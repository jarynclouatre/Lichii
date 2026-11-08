// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_oot_object_motion.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {

OotObjectMotion* instance{};

void MemoryBarrier(vk::CommandBuffer& cmdbuf) {
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           0, barrier);
}

// EDEN_OOT_OBJECT_MOTION_REPEATED=1: also give history to meshes drawn more than once a frame,
// matched by draw order (which the game does not keep: copies trade histories).
bool KeepRepeated() {
    static const bool keep = [] {
        const char* value = std::getenv("EDEN_OOT_OBJECT_MOTION_REPEATED");
        return value != nullptr && value[0] == '1';
    }();
    return keep;
}

u64 Mix(u64 hash, u64 value) {
    return (hash ^ value) * 0x100000001b3ULL + (hash >> 29);
}

} // Anonymous namespace

OotObjectMotion::OotObjectMotion(const Device& device_, MemoryAllocator& allocator_)
    : device{device_}, allocator{allocator_} {
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (device.IsBufferDeviceAddressSupported()) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    history = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = HistoryBytes,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::DeviceLocal);
    if (device.IsBufferDeviceAddressSupported()) {
        history_address = device.GetLogical().GetBufferDeviceAddress(*history);
    }
    instance = this;
    LOG_INFO(Render_Vulkan, "OoT DLSS: object motion history {} MiB", HistoryBytes >> 20);
}

OotObjectMotion::~OotObjectMotion() {
    if (instance == this) {
        instance = nullptr;
    }
}

OotObjectMotion* OotObjectMotion::Get() noexcept {
    return instance;
}

void OotObjectMotion::EnsureImage(Scheduler& scheduler, VkExtent2D size) {
    if (!history_initialized) {
        history_initialized = true;
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([buffer = *history](vk::CommandBuffer cmdbuf) {
            cmdbuf.FillBuffer(buffer, 0, VK_WHOLE_SIZE, 0);
            MemoryBarrier(cmdbuf);
        });
    }
    if (pixels && extent.width == size.width && extent.height == size.height) {
        return;
    }
    if (pixels) {
        scheduler.Finish(); // pending draws may still write the old buffer
    }
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (device.IsBufferDeviceAddressSupported()) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    pixels = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = VkDeviceSize{size.width} * size.height * 8,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::DeviceLocal);
    pixels_address =
        device.IsBufferDeviceAddressSupported() ? device.GetLogical().GetBufferDeviceAddress(*pixels) : 0;
    extent = size;
    ClearPixels(scheduler);
    LOG_INFO(Render_Vulkan, "OoT DLSS: object motion buffer for {}x{}", size.width, size.height);
}

void OotObjectMotion::ClearPixels(Scheduler& scheduler) {
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([buffer = *pixels](vk::CommandBuffer cmdbuf) {
        MemoryBarrier(cmdbuf);
        cmdbuf.FillBuffer(buffer, 0, VK_WHOLE_SIZE, ~0u); // no fragment
        MemoryBarrier(cmdbuf);
    });
}

OotObjectMotion::Layout OotObjectMotion::Allocate(u64 identity, u32 range, u32 instances,
                                                  u32 vertex_bias, u32 instance_bias,
                                                  bool by_order) {
    ++stats.draws;
    drawn = true;
    u32& occurrence = occurrences[identity];
    stats.repeated += occurrence != 0 ? 1 : 0;
    const auto previous_count = previous_occurrences.find(identity);
    const bool repeated = occurrence != 0 || (previous_count != previous_occurrences.end() &&
                                              previous_count->second > 1);
    const u64 key = Mix(identity, occurrence++);
    const u64 slots = u64{range} * instances;
    if (slots == 0 || used + slots > SlotsPerHalf) {
        ++stats.overflow;
        return Layout{.flags = 0};
    }
    const u32 base = half * SlotsPerHalf + used;
    used += static_cast<u32>(slots);
    stats.slots += slots;
    current[key] = Slot{base, range, instances};
    u32 previous_base = ~0u;
    if (const auto it = previous.find(key);
        (!repeated || by_order || KeepRepeated()) && it != previous.end() &&
        it->second.range == range &&
        it->second.instances == instances) {
        previous_base = it->second.base;
        ++stats.matched;
    }
    return Layout{
        .current_base = base,
        .previous_base = previous_base,
        .vertex_bias = vertex_bias,
        .vertex_range = range,
        .instance_bias = instance_bias,
        .flags = 1,
    };
}

void OotObjectMotion::BeginFrame(Scheduler& scheduler) {
    if (!drawn) {
        // A frame without scene draws (menus, a skipped frame) keeps last frame's history.
        return;
    }
    drawn = false;
    previous = std::move(current);
    current.clear();
    previous_occurrences = std::move(occurrences);
    occurrences.clear();
    half ^= 1;
    used = 0;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([](vk::CommandBuffer cmdbuf) { MemoryBarrier(cmdbuf); });
    if (!std::exchange(combined, false) && pixels) {
        ClearPixels(scheduler);
    }
    if (++stats.frames % 600 == 0) {
        LOG_INFO(Render_Vulkan,
                 "OoT DLSS object motion, last 600 frames: {} draws, {} with history, {} repeated "
                 "identities, {} overflowed, {} slots per frame",
                 stats.draws, stats.matched, stats.repeated, stats.overflow, stats.slots / 600);
        stats = Stats{.frames = stats.frames};
    }
}

} // namespace Vulkan
