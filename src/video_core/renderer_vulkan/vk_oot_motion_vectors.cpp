// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "common/logging.h"

#include "video_core/host_shaders/oot_motion_vectors_comp_spv.h"
#include "video_core/host_shaders/oot_particle_composite_comp_spv.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_oot_motion_vectors.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {

constexpr VkFormat Format = VK_FORMAT_R16G16_SFLOAT;

struct PushConstants {
    std::array<f32, 16> reproject_rows;
    std::array<f32, 2> render_size;
    std::array<f32, 2> jitter;
    f32 flip_y;
    u32 object_flags;
    f32 mask_gain;
    f32 fog_gain;
    f32 camera_deadzone;
};

bool DebugRequested() {
    static const bool requested = [] {
        const char* value = std::getenv("EDEN_OOT_MOTION_DEBUG");
        return value != nullptr && value[0] == '1';
    }();
    return requested;
}

/// Camera motion below this many render pixels per axis is dropped: a still camera's matrix
/// wobbles by ~0.1 pixel. EDEN_OOT_CAMERA_DEADZONE=<pixels> (0.35 by default; 0 keeps it all).
f32 CameraDeadzone() {
    static const f32 deadzone = [] {
        const char* value = std::getenv("EDEN_OOT_CAMERA_DEADZONE");
        return value != nullptr ? static_cast<f32>(std::strtod(value, nullptr)) : 0.35f;
    }();
    return deadzone;
}

/// EDEN_OOT_MOTION_CHECK=1: measure how well the motion predicts each frame from the last
/// (reported with the statistics).
bool CheckRequested() {
    static const bool requested = [] {
        const char* value = std::getenv("EDEN_OOT_MOTION_CHECK");
        return value != nullptr && value[0] == '1';
    }();
    return requested;
}

/// EDEN_OOT_OBJECT_MOTION_STATS=<n>: log the motion statistics every n evaluations (0: off).
u32 StatsInterval() {
    static const u32 interval = [] {
        const char* value = std::getenv("EDEN_OOT_OBJECT_MOTION_STATS");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0U;
    }();
    return interval;
}

bool StatsRequested() {
    return StatsInterval() != 0 || CheckRequested();
}

vk::Buffer CreateStorageBuffer(MemoryAllocator& allocator, VkDeviceSize size, MemoryUsage usage) {
    return allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        usage);
}

// Everything before (the scene's depth writes) is visible to the pass, and the pass's writes to
// everything after (DLSS reads them through Streamline).
void FullBarrier(vk::CommandBuffer& cmdbuf) {
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           0, barrier);
}

} // Anonymous namespace

OotMotionVectors::OotMotionVectors(const Device& device_, MemoryAllocator& allocator_)
    : device{device_}, allocator{allocator_} {
    descriptor_pool = CreateWrappedDescriptorPool(
        device, 5 * SetCount, SetCount,
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER});
    descriptor_set_layout = CreateWrappedDescriptorSetLayout(
        device,
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        VK_SHADER_STAGE_COMPUTE_BIT);
    const std::array layouts{*descriptor_set_layout, *descriptor_set_layout,
                             *descriptor_set_layout, *descriptor_set_layout};
    descriptor_sets = CreateWrappedDescriptorSets(descriptor_pool, layouts);
    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    pipeline_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = descriptor_set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
    shader = CreateWrappedShaderModule(device, OOT_MOTION_VECTORS_COMP_SPV);
    pipeline = CreateWrappedComputePipeline(device, pipeline_layout, *shader);
    sampler = CreateWrappedSampler(device, VK_FILTER_NEAREST);
    linear_sampler = CreateWrappedSampler(device, VK_FILTER_LINEAR);
    composite_pool = CreateWrappedDescriptorPool(
        device, 3 * SetCount, SetCount,
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});
    composite_set_layout = CreateWrappedDescriptorSetLayout(
        device,
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
        VK_SHADER_STAGE_COMPUTE_BIT);
    const std::array composite_layouts{*composite_set_layout, *composite_set_layout,
                                       *composite_set_layout, *composite_set_layout};
    composite_sets = CreateWrappedDescriptorSets(composite_pool, composite_layouts);
    const VkPushConstantRange composite_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(std::array<f32, 2>),
    };
    composite_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = composite_set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &composite_range,
    });
    composite_shader = CreateWrappedShaderModule(device, OOT_PARTICLE_COMPOSITE_COMP_SPV);
    composite_pipeline = CreateWrappedComputePipeline(device, composite_layout, *composite_shader);
    dummy_pixels = CreateStorageBuffer(allocator, 64, MemoryUsage::DeviceLocal);
    stats = CreateStorageBuffer(allocator, 64, MemoryUsage::Download);
    // 11 statistics words and 4 check words.
    if (auto mapped = stats.Mapped(); !mapped.empty()) {
        std::memset(mapped.data(), 0, mapped.size());
        stats.Flush();
    }
}

// Logs the counts since the last report (they lag the GPU by a frame or two).
void OotMotionVectors::ReportStats() {
    const auto mapped = stats.Mapped();
    if (mapped.size() < sizeof(reported)) {
        return;
    }
    stats.Invalidate();
    std::array<u32, 15> now{};
    std::memcpy(now.data(), mapped.data(), sizeof(now));
    std::array<u32, 15> delta{};
    for (size_t i = 0; i < now.size(); ++i) {
        delta[i] = now[i] - reported[i];
    }
    reported = now;
    const auto percent = [](u32 part, u32 whole) {
        return whole == 0 ? 0.0 : 100.0 * part / whole;
    };
    LOG_INFO(Render_Vulkan,
             "OoT DLSS object motion: {:.1f}% of scene pixels written, {:.1f}% used; of those "
             "{:.1f}% within 1 px of the camera motion (y as is), {:.1f}% (y negated); written behind "
             "the scene depth {:.1f}%, in front {:.1f}%",
             percent(delta[1], delta[0]), percent(delta[2], delta[0]),
             percent(delta[3], delta[2]), percent(delta[4], delta[2]),
             percent(delta[5], delta[0]), percent(delta[6], delta[0]));
    LOG_INFO(Render_Vulkan,
             "OoT DLSS object motion: mean |x| object {:.2f} px, camera {:.2f} px; x direction "
             "agrees on {:.1f}% of {} moving pixels",
             delta[2] == 0 ? 0.0 : delta[7] / 16.0 / delta[2],
             delta[2] == 0 ? 0.0 : delta[8] / 16.0 / delta[2], percent(delta[9], delta[10]),
             delta[10]);
    if (CheckRequested() && delta[14] != 0) {
        LOG_INFO(Render_Vulkan,
                 "OoT DLSS motion check: mean luminance error predicting from last frame with the "
                 "motion {:.4f}, y negated {:.4f}, no motion {:.4f} ({} moving samples)",
                 delta[11] / 1024.0 / delta[14], delta[12] / 1024.0 / delta[14],
                 delta[13] / 1024.0 / delta[14], delta[14]);
    }
}

OotMotionVectors::~OotMotionVectors() = default;

void OotMotionVectors::SnapshotScene(Scheduler& scheduler, VkImage source, VkFormat format,
                                     VkExtent2D size, u32 which, bool restore) {
    if (restore) {
        if (!scene_images[which] || scene_formats[which] != format ||
            scene_extents[which].width != size.width ||
            scene_extents[which].height != size.height) {
            return;
        }
        const VkImage copy_image = *scene_images[which];
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([=](vk::CommandBuffer cmdbuf) {
            FullBarrier(cmdbuf);
            const VkImageCopy copy{
                .srcSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .srcOffset{},
                .dstSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .dstOffset{},
                .extent{size.width, size.height, 1},
            };
            cmdbuf.CopyImage(copy_image, VK_IMAGE_LAYOUT_GENERAL, source,
                             VK_IMAGE_LAYOUT_GENERAL, copy);
            FullBarrier(cmdbuf);
        });
        return;
    }
    bool initialize = false;
    if (!scene_images[which] || scene_formats[which] != format ||
        scene_extents[which].width != size.width || scene_extents[which].height != size.height) {
        if (scene_images[which]) {
            scheduler.Finish(); // the old copy may still be read
        }
        scene_formats[which] = format;
        scene_images[which] = allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {size.width, size.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        scene_views[which] = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *scene_images[which],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components{},
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });
        scene_extents[which] = size;
        initialize = true;
    }
    const VkImage destination = *scene_images[which];
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        if (initialize) {
            TransitionImageLayout(cmdbuf, destination, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_UNDEFINED);
        }
        FullBarrier(cmdbuf);
        const VkImageCopy copy{
            .srcSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset{},
            .dstSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset{},
            .extent{size.width, size.height, 1},
        };
        cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_GENERAL, destination, VK_IMAGE_LAYOUT_GENERAL,
                         copy);
        FullBarrier(cmdbuf);
    });
}

bool OotMotionVectors::CompositeParticles(Scheduler& scheduler, VkImageView output_view,
                                          VkExtent2D output_size, VkImageView without_view,
                                          VkExtent2D render_size) {
    if (!scene_images[2] || scene_extents[2].width != render_size.width ||
        scene_extents[2].height != render_size.height) {
        return false;
    }
    const u32 set_index = next_composite_set;
    next_composite_set = (next_composite_set + 1) % SetCount;
    scheduler.Wait(composite_ticks[set_index]);
    const VkDescriptorSet set = composite_sets[set_index];
    const VkDescriptorImageInfo output_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = output_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo with_info{
        .sampler = *linear_sampler,
        .imageView = *scene_views[2],
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo without_info{
        .sampler = *linear_sampler,
        .imageView = without_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const auto write = [&](u32 binding, VkDescriptorType type, const VkDescriptorImageInfo* info) {
        return VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = type,
            .pImageInfo = info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
    };
    const std::array writes{
        write(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &output_info),
        write(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &with_info),
        write(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &without_info),
    };
    device.GetLogical().UpdateDescriptorSets(writes, {});
    const std::array<f32, 2> constants{static_cast<f32>(output_size.width),
                                       static_cast<f32>(output_size.height)};
    const VkPipeline compute_pipeline = *composite_pipeline;
    const VkPipelineLayout layout = *composite_layout;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        FullBarrier(cmdbuf);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, set, {});
        cmdbuf.PushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((output_size.width + 7) / 8, (output_size.height + 7) / 8, 1);
        FullBarrier(cmdbuf);
    });
    composite_ticks[set_index] = scheduler.CurrentTick();
    return true;
}

void OotMotionVectors::CreateImage(VkExtent2D size) {
    image = allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = Format,
        .extent = {size.width, size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = Format,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    mask_image = allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R16_SFLOAT,
        .extent = {size.width, size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    mask_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *mask_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R16_SFLOAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    const auto colour_image = [&](VkExtent2D extent_) {
        return allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R16G16B16A16_SFLOAT,
            .extent = {extent_.width, extent_.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
    };
    const auto colour_view = [&](VkImage target) {
        return device.GetLogical().CreateImageView(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = target,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R16G16B16A16_SFLOAT,
            .components{},
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });
    };
    // Last frame's colour for the motion check (1x1 stand-ins without it).
    const VkExtent2D history_size = CheckRequested() ? size : VkExtent2D{1, 1};
    for (size_t i = 0; i < history_images.size(); ++i) {
        history_images[i] = colour_image(history_size);
        history_views[i] = colour_view(*history_images[i]);
    }
    history_valid = false;
    extent = size;
    image_initialized = false;
}

VkImageView OotMotionVectors::Generate(Scheduler& scheduler, VkImageView depth_view,
                                       VkExtent2D size, const std::array<f32, 16>& reproject,
                                       std::array<f32, 2> jitter, bool flip_y,
                                       VkBuffer object_pixels, VkImageView effects_view,
                                       f32 mask_gain, VkImageView fog_view, f32 fog_gain,
                                       VkImageView color_view) {
    if (!image || extent.width != size.width || extent.height != size.height) {
        if (image) {
            scheduler.Finish(); // the old image may still be read
        }
        CreateImage(size);
    }
    const u32 set_index = next_set;
    next_set = (next_set + 1) % SetCount;
    scheduler.Wait(set_ticks[set_index]);
    const VkDescriptorSet set = descriptor_sets[set_index];
    const VkDescriptorImageInfo depth_info{
        .sampler = *sampler,
        .imageView = depth_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo output_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = *view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorBufferInfo object_info{
        .buffer = object_pixels != VK_NULL_HANDLE ? object_pixels : *dummy_pixels,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    const VkDescriptorBufferInfo stats_info{
        .buffer = *stats,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    // Without an effects layer the depth stands in (the shader skips it).
    const VkDescriptorImageInfo effects_info{
        .sampler = *linear_sampler,
        .imageView = effects_view != VK_NULL_HANDLE ? effects_view : depth_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo fog_info{
        .sampler = *linear_sampler,
        .imageView = fog_view != VK_NULL_HANDLE ? fog_view : depth_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const bool check = CheckRequested() && color_view != VK_NULL_HANDLE;
    const VkDescriptorImageInfo current_info{
        .sampler = *sampler,
        .imageView = check ? color_view : depth_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo previous_info{
        .sampler = *linear_sampler,
        .imageView = *history_views[history_index],
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo next_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = *history_views[history_index ^ 1],
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo mask_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = *mask_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const std::array writes{
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &depth_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &output_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &object_info,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &stats_info,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 4,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &effects_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 5,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &mask_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 6,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &fog_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 7,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &current_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 8,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &previous_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 9,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &next_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
    };
    device.GetLogical().UpdateDescriptorSets(writes, {});

    const PushConstants constants{
        .reproject_rows = reproject,
        .render_size = {static_cast<f32>(size.width), static_cast<f32>(size.height)},
        .jitter = jitter,
        .flip_y = flip_y ? 1.0f : 0.0f,
        .object_flags = (object_pixels != VK_NULL_HANDLE ? 1u : 0u) |
                        (object_pixels != VK_NULL_HANDLE && StatsRequested() ? 2u : 0u) |
                        (DebugRequested() ? 4u : 0u) |
                        (effects_view != VK_NULL_HANDLE ? 8u : 0u) |
                        (fog_view != VK_NULL_HANDLE ? 16u : 0u) | (check ? 32u : 0u) |
                        (check && history_valid ? 64u : 0u),
        .mask_gain = mask_gain,
        .fog_gain = fog_gain,
        .camera_deadzone = CameraDeadzone(),
    };
    // The 32-bit sums of 1/16 pixels wrap within a few dozen 4K frames: keep n small.
    if (StatsRequested() && ++generated % (StatsInterval() != 0 ? StatsInterval() : 30) == 0) {
        ReportStats();
    }
    const bool initialize = !std::exchange(image_initialized, true);

    const VkImage output_image = *image;
    const VkImage mask = *mask_image;
    const VkImage history_a = *history_images[0];
    const VkImage history_b = *history_images[1];
    if (check) {
        history_index ^= 1;
        history_valid = true;
    }
    const VkPipeline compute_pipeline = *pipeline;
    const VkPipelineLayout layout = *pipeline_layout;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        if (initialize) {
            TransitionImageLayout(cmdbuf, output_image, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_UNDEFINED);
            TransitionImageLayout(cmdbuf, mask, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_UNDEFINED);
            TransitionImageLayout(cmdbuf, history_a, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_UNDEFINED);
            TransitionImageLayout(cmdbuf, history_b, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_UNDEFINED);
        }

        FullBarrier(cmdbuf);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, set, {});
        cmdbuf.PushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
        FullBarrier(cmdbuf);
    });
    set_ticks[set_index] = scheduler.CurrentTick();
    return *view;
}

} // namespace Vulkan
