// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA Streamline) in place of a game's DLSS evaluation.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <utility>
#include <string_view>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/dlss_evaluation.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
// Streamline clears the output before DLSS writes it.
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkFormat EXPOSURE_FORMAT = VK_FORMAT_R32_SFLOAT;
constexpr VkImageUsageFlags EXPOSURE_USAGE =
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

int EnvInt(const char* name, int fallback) {
    const char* const value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

// DLSS validates the render size against the mode's range: pick the mode for this ratio.
sl::DLSSMode ModeFor(VkExtent2D render, VkExtent2D upscale) {
    const f32 ratio = static_cast<f32>(render.width) / static_cast<f32>(upscale.width);
    if (ratio >= 0.99f) {
        return sl::DLSSMode::eDLAA;
    }
    if (ratio >= 0.65f) {
        return sl::DLSSMode::eMaxQuality;
    }
    if (ratio >= 0.57f) {
        return sl::DLSSMode::eBalanced;
    }
    if (ratio >= 0.49f) {
        return sl::DLSSMode::eMaxPerformance;
    }
    return sl::DLSSMode::eUltraPerformance;
}

// Preset K for every mode: NVIDIA's defaults for Performance/Ultra Performance (M, L) draw thin
// horizontal lines in Fire Emblem at 2x resolution scale. EDEN_NX2_DLSS_PRESET=J|K|L|M picks
// another model, EDEN_NX2_DLSS_PRESET=default leaves the choice to NVIDIA.
sl::DLSSPreset Preset() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_PRESET");
    if (value && value[0] >= 'J' && value[0] <= 'M') {
        return static_cast<sl::DLSSPreset>(static_cast<u32>(sl::DLSSPreset::ePresetJ) +
                                           static_cast<u32>(value[0] - 'J'));
    }
    if (value && std::string_view{value} == "default") {
        return sl::DLSSPreset::eDefault;
    }
    return sl::DLSSPreset::ePresetK;
}

sl::float4x4 Identity() {
    sl::float4x4 matrix;
    matrix.setRow(0, {1.0f, 0.0f, 0.0f, 0.0f});
    matrix.setRow(1, {0.0f, 1.0f, 0.0f, 0.0f});
    matrix.setRow(2, {0.0f, 0.0f, 1.0f, 0.0f});
    matrix.setRow(3, {0.0f, 0.0f, 0.0f, 1.0f});
    return matrix;
}

/// Row-major, for column vectors (the game's convention).
using Matrix = std::array<f32, 16>;

Matrix Multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            f32 sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a[row * 4 + k] * b[k * 4 + column];
            }
            result[row * 4 + column] = sum;
        }
    }
    return result;
}

std::optional<Matrix> Invert(const Matrix& m) {
    // Gauss-Jordan with partial pivoting, in double.
    std::array<std::array<double, 8>, 4> a{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            a[row][column] = m[row * 4 + column];
        }
        a[row][4 + row] = 1.0;
    }
    for (int column = 0; column < 4; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 4; ++row) {
            if (std::abs(a[row][column]) > std::abs(a[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(a[pivot][column]) < 1e-12) {
            return std::nullopt;
        }
        std::swap(a[column], a[pivot]);
        const double scale = 1.0 / a[column][column];
        for (double& value : a[column]) {
            value *= scale;
        }
        for (int row = 0; row < 4; ++row) {
            if (row != column) {
                const double factor = a[row][column];
                for (int k = 0; k < 8; ++k) {
                    a[row][k] -= factor * a[column][k];
                }
            }
        }
    }
    Matrix result{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            result[row * 4 + column] = static_cast<f32>(a[row][4 + column]);
        }
    }
    return result;
}

/// Streamline's matrices are row-major for row vectors: the transpose of ours.
sl::float4x4 ToStreamline(const Matrix& m) {
    sl::float4x4 matrix;
    for (int row = 0; row < 4; ++row) {
        matrix.setRow(row, {m[row], m[4 + row], m[8 + row], m[12 + row]});
    }
    return matrix;
}

/// Frame generation's camera constants from the game's OpenGL-style matrices. Streamline's clip
/// space has z = depth in [0, 1] and y up on screen; the game's NDC has z = depth * 2 - 1 and y
/// as the image rows go, upward only with flip_y. Returns false when the matrices don't look
/// like a perspective camera (the constants then keep their identities).
bool SetCameraConstants(sl::Constants& constants,
                        const StreamlineDlssUpscaler::Inputs::Camera& camera, f32 aspect) {
    const f32 y = camera.flip_y ? 1.0f : -1.0f;
    const Matrix to_game{1, 0, 0, 0, 0, y, 0, 0, 0, 0, 2, -1, 0, 0, 0, 1};
    const Matrix from_game{1, 0, 0, 0, 0, y, 0, 0, 0, 0, 0.5f, 0.5f, 0, 0, 0, 1};
    const Matrix clip_to_previous = Multiply(from_game, Multiply(camera.reproject, to_game));
    const std::optional<Matrix> previous_to_clip = Invert(clip_to_previous);
    if (!previous_to_clip) {
        return false;
    }
    // A perspective view-projection: row 3 is minus the view's z axis (clip w = -z), rows 0 and
    // 1 the view's x and y axes scaled by the projection, row 2 a mix of row 3 and a constant.
    const f32* m = camera.view_projection.data();
    const auto length = [](const f32* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    const f32 scale_x = length(m);
    const f32 scale_y = length(m + 4);
    const f32 scale_w = length(m + 12);
    if (scale_x < 1e-4f || scale_y < 1e-4f || scale_w < 1e-4f) {
        return false;
    }
    std::array<f32, 4> view_z{};
    for (int i = 0; i < 4; ++i) {
        view_z[i] = -m[12 + i] / scale_w;
    }
    const f32 a = m[8] * view_z[0] + m[9] * view_z[1] + m[10] * view_z[2];
    const f32 b = m[11] - a * view_z[3];
    const f32 near_plane = b / (a - 1.0f);
    const f32 far_plane = b / (a + 1.0f);
    const bool planes_valid = std::isfinite(near_plane) && std::isfinite(far_plane) &&
                              near_plane > 0.0f && far_plane > near_plane;
    const Matrix projection{scale_x, 0, 0, 0, 0, scale_y, 0, 0, 0, 0, a, b, 0, 0, -1, 0};
    const Matrix view_to_clip = Multiply(from_game, projection);
    const std::optional<Matrix> clip_to_view = Invert(view_to_clip);
    if (!clip_to_view) {
        return false;
    }
    constants.cameraViewToClip = ToStreamline(view_to_clip);
    constants.clipToCameraView = ToStreamline(*clip_to_view);
    constants.clipToPrevClip = ToStreamline(clip_to_previous);
    constants.prevClipToClip = ToStreamline(*previous_to_clip);
    const sl::float3 right{m[0] / scale_x, m[1] / scale_x, m[2] / scale_x};
    const sl::float3 up{m[4] / scale_y, m[5] / scale_y, m[6] / scale_y};
    const sl::float3 forward{-view_z[0], -view_z[1], -view_z[2]};
    const f32 tx = m[3] / scale_x;
    const f32 ty = m[7] / scale_y;
    const f32 tz = view_z[3];
    constants.cameraRight = right;
    constants.cameraUp = up;
    constants.cameraFwd = forward;
    constants.cameraPos = {-(tx * right.x + ty * up.x + tz * view_z[0]),
                           -(tx * right.y + ty * up.y + tz * view_z[1]),
                           -(tx * right.z + ty * up.z + tz * view_z[2])};
    if (planes_valid) {
        constants.cameraNear = near_plane;
        constants.cameraFar = far_plane;
    }
    constants.cameraFOV = 2.0f * std::atan(1.0f / scale_y);
    constants.cameraAspectRatio = aspect;
    if (static u32 calls{}; calls++ % 1800 == 0) {
        LOG_INFO(Render_Vulkan,
                 "DLSS camera: near {:.3f}, far {:.1f}{}, vertical FOV {:.1f} degrees, position "
                 "({:.1f}, {:.1f}, {:.1f})",
                 near_plane, far_plane, planes_valid ? "" : " (not used)",
                 constants.cameraFOV * 57.2958f, constants.cameraPos.x, constants.cameraPos.y,
                 constants.cameraPos.z);
    }
    return true;
}

VkImageMemoryBarrier Barrier(const StreamlineDlssUpscaler::Image& image, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = range,
    };
}

sl::Resource Resource(VkImage image, VkImageView view, VkFormat format, VkImageLayout layout,
                      VkExtent2D size, VkImageUsageFlags usage) {
    sl::Resource resource{sl::ResourceType::eTex2d, image, nullptr, view,
                          static_cast<uint32_t>(layout)};
    resource.width = size.width;
    resource.height = size.height;
    resource.nativeFormat = static_cast<uint32_t>(format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0;
    resource.usage = usage;
    return resource;
}

} // Anonymous namespace

StreamlineDlssUpscaler::StreamlineDlssUpscaler(const Device& device_, Scheduler& scheduler_,
                                               MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (!Streamline::IsDlssAvailable()) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    const sl::Result result = Streamline::GetApi().get_feature_function(
        sl::kFeatureDLSS, "slDLSSSetOptions", set_options);
    if (result != sl::Result::eOk || !set_options) {
        LOG_ERROR(Render_Vulkan, "Streamline's DLSS has no slDLSSSetOptions (result {})",
                  static_cast<int>(result));
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    point_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

StreamlineDlssUpscaler::~StreamlineDlssUpscaler() {
    // Not scheduler.Finish(): this runs when the renderer goes away, after the game's channels are
    // released, and a submit would touch their state. Nothing more gets recorded by then.
    void(device.GetLogical().WaitIdle());
    if (Streamline::IsDlssAvailable()) {
        Streamline::GetApi().free_resources(sl::kFeatureDLSS, sl::ViewportHandle{0u});
    }
}

void StreamlineDlssUpscaler::CreateExposure() {
    exposure_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = EXPOSURE_FORMAT,
        .extent = {1, 1, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = EXPOSURE_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    exposure_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *exposure_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = EXPOSURE_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
}

void StreamlineDlssUpscaler::ClearExposure(vk::CommandBuffer cmdbuf, VkImage image, f32 value) {
    const VkImageSubresourceRange range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    // Its old contents are not needed: from undefined (the last evaluation's reads done) to
    // GENERAL, cleared, then readable by DLSS.
    const auto barrier = [&](VkAccessFlags src, VkAccessFlags dst, VkImageLayout old_layout) {
        return VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = src,
            .dstAccessMask = dst,
            .oldLayout = old_layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                           barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_UNDEFINED));
    cmdbuf.ClearColorImage(image, VK_IMAGE_LAYOUT_GENERAL, VkClearColorValue{.float32{value, value, value, value}},
                           range);
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                           barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                   VK_IMAGE_LAYOUT_GENERAL));
}

void StreamlineDlssUpscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    const auto create = [&](vk::Image& output_image, vk::ImageView& output_view) {
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               VkImageMemoryBarrier{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = 0,
                                   .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange{
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1,
                                   },
                               });
    });
    };
    create(output_image, output_view);
}

bool StreamlineDlssUpscaler::Upscale(const Inputs& in) {
    if (failed) {
        return false;
    }
    bool reset = in.reset;
    bool options_changed = false;
    if (!output_image || in.render_size.width != render_extent.width ||
        in.render_size.height != render_extent.height ||
        in.upscale_size.width != upscale_extent.width ||
        in.upscale_size.height != upscale_extent.height) {
        CreateOutput(in.upscale_size);
        render_extent = in.render_size;
        upscale_extent = in.upscale_size;
        options_changed = true;
        reset = true;
    }

    // The game's own setup (Fortune's Weave: render-size motion vectors, reversed depth). Its
    // colour is linear and goes above 1 with no exposure passed, so by default DLSS gets the HDR
    // path with its own auto exposure. EDEN_NX2_DLSS_HDR=0 / EDEN_NX2_DLSS_AUTO_EXPOSURE=0 turn
    // those off.
    sl::DLSSOptions options{};
    options.mode = ModeFor(in.render_size, in.upscale_size);
    options.outputWidth = in.upscale_size.width;
    options.outputHeight = in.upscale_size.height;
    options.colorBuffersHDR =
        (in.color_hdr >= 0 ? in.color_hdr : EnvInt("EDEN_NX2_DLSS_HDR", 1)) != 0 ? sl::eTrue
                                                                                 : sl::eFalse;
    options.useAutoExposure = in.color_hdr != 0 && !in.exposure &&
                                      EnvInt("EDEN_NX2_DLSS_AUTO_EXPOSURE", 1) != 0
                                  ? sl::eTrue
                                  : sl::eFalse;
    if (const bool automatic = options.useAutoExposure == sl::eTrue;
        automatic != applied_auto_exposure) {
        applied_auto_exposure = automatic;
        options_changed = true;
        LOG_INFO(Render_Vulkan, "DLSS: {} exposure", automatic ? "auto" : in.exposure ? "the game's" : "no");
    }
    if (in.exposure && !exposure_image) {
        CreateExposure();
    }
    // The DLSS preset setting, else the caller's choice, else Preset() (EDEN_NX2_DLSS_PRESET).
    int chosen = in.preset;
    switch (Settings::values.dlss_preset.GetValue()) {
    case Settings::DlssPreset::K:
        chosen = static_cast<int>(sl::DLSSPreset::ePresetK);
        break;
    case Settings::DlssPreset::L:
        chosen = static_cast<int>(sl::DLSSPreset::ePresetL);
        break;
    case Settings::DlssPreset::M:
        chosen = static_cast<int>(sl::DLSSPreset::ePresetM);
        break;
    default:
        break;
    }
    const sl::DLSSPreset preset = chosen >= 0 && std::getenv("EDEN_NX2_DLSS_PRESET") == nullptr
                                      ? static_cast<sl::DLSSPreset>(chosen)
                                      : Preset();
    if (static_cast<u32>(preset) != applied_preset_value) {
        applied_preset_value = static_cast<u32>(preset);
        options_changed = true;
    }
    options.dlaaPreset = preset;
    options.qualityPreset = preset;
    options.balancedPreset = preset;
    options.performancePreset = preset;
    options.ultraPerformancePreset = preset;
    options.ultraQualityPreset = preset;
    if (options_changed) {
        LOG_INFO(Render_Vulkan, "NVIDIA DLSS (Streamline): {}x{} -> {}x{}, mode {}, preset {}",
                 in.render_size.width, in.render_size.height, in.upscale_size.width,
                 in.upscale_size.height, static_cast<int>(options.mode),
                 static_cast<int>(preset));
    }

    const u32 index = frame_index++;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, reset, options, options_changed, index,
                      output_image_handle = *output_image,
                      output_view_handle = *output_view,
                      exposure_image_handle = exposure_image ? *exposure_image : VkImage{},
                      exposure_view_handle = exposure_view ? *exposure_view : VkImageView{}](
                         vk::CommandBuffer cmdbuf) {
        const Streamline::Api& api = Streamline::GetApi();
        const sl::ViewportHandle viewport{0u};
        // The caller draws the output this frame even when DLSS writes nothing to it: clear it
        // then, rather than show undefined contents (FSR 3.1 takes over from the next frame).
        const auto clear_output = [&] {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            const auto barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = output_image_handle,
                    .subresourceRange = range,
                };
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                   barrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                           VK_ACCESS_TRANSFER_WRITE_BIT));
            cmdbuf.ClearColorImage(output_image_handle, VK_IMAGE_LAYOUT_GENERAL,
                                   VkClearColorValue{}, range);
            cmdbuf.PipelineBarrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
        };
        sl::FrameToken* frame{};

        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs come from the game's draws and dispatches and stay in GENERAL otherwise;
        // DLSS reads them as sampled images.
        const std::array to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };

        // DLSS super resolution only reads the jitter, motion and depth conventions; the camera
        // fields are for frame generation, filled from the game's matrices when the caller has
        // them.
        const f32 rw = static_cast<f32>(in.render_size.width);
        const f32 rh = static_cast<f32>(in.render_size.height);
        sl::Constants constants{};
        constants.cameraViewToClip = Identity();
        constants.clipToCameraView = Identity();
        constants.clipToLensClip = Identity();
        constants.clipToPrevClip = Identity();
        constants.prevClipToClip = Identity();
        constants.jitterOffset = {in.jitter[0], in.jitter[1]};
        // Streamline wants motion in [-1,1] screen units; ours turns into render pixels.
        constants.mvecScale = {in.motion_vector_scale[0] / rw, in.motion_vector_scale[1] / rh};
        constants.cameraPinholeOffset = {0.0f, 0.0f};
        constants.cameraPos = {0.0f, 0.0f, 0.0f};
        constants.cameraUp = {0.0f, 1.0f, 0.0f};
        constants.cameraRight = {1.0f, 0.0f, 0.0f};
        constants.cameraFwd = {0.0f, 0.0f, 1.0f};
        constants.cameraNear = 0.1f;
        constants.cameraFar = 1000.0f;
        constants.cameraFOV = 1.0f;
        constants.cameraAspectRatio = rw / rh;
        constants.depthInverted = in.depth_inverted ? sl::eTrue : sl::eFalse;
        constants.cameraMotionIncluded = sl::eTrue;
        constants.motionVectors3D = sl::eFalse;
        constants.reset = reset ? sl::eTrue : sl::eFalse;
        constants.motionVectorsJittered = sl::eFalse;
        if (in.camera && !SetCameraConstants(constants, *in.camera, rw / rh)) {
            static bool logged{};
            if (!std::exchange(logged, true)) {
                LOG_WARNING(Render_Vulkan, "DLSS: the game's camera matrices are not invertible; "
                                           "frame generation gets identity matrices");
            }
        }

        sl::Resource color = Resource(in.color.image, in.color.view, in.color.format,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                      VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource depth = Resource(in.depth.image, in.depth.view, in.depth.format,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                      VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource motion = Resource(in.motion_vectors.image, in.motion_vectors.view,
                                       in.motion_vectors.format,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, in.render_size,
                                       VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource output = Resource(output_image_handle, output_view_handle, OUTPUT_FORMAT,
                                       VK_IMAGE_LAYOUT_GENERAL, in.upscale_size, OUTPUT_USAGE);
        const sl::Extent render_area{0, 0, in.render_size.width, in.render_size.height};
        const sl::Extent output_area{0, 0, in.upscale_size.width, in.upscale_size.height};
        // Local tags: everything stays put until the evaluation is recorded.
        sl::ResourceTag color_tag{&color, sl::kBufferTypeScalingInputColor,
                                  sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag depth_tag{&depth, sl::kBufferTypeDepth,
                                  sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag motion_tag{&motion, sl::kBufferTypeMotionVectors,
                                   sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        sl::ResourceTag output_tag{&output, sl::kBufferTypeScalingOutputColor,
                                   sl::ResourceLifecycle::eValidUntilEvaluate, &output_area};
        sl::Resource mask{};
        sl::ResourceTag mask_tag{};
        if (in.bias_mask) {
            mask = Resource(in.bias_mask->image, in.bias_mask->view, in.bias_mask->format,
                            VK_IMAGE_LAYOUT_GENERAL, in.render_size,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            mask_tag = sl::ResourceTag{&mask, sl::kBufferTypeBiasCurrentColorHint,
                                       sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
        }
        std::array<const sl::BaseStructure*, 7> inputs{&viewport, &color_tag, &depth_tag,
                                                       &motion_tag, &output_tag};
        u32 input_count = 5;
        if (in.bias_mask) {
            inputs[input_count++] = &mask_tag;
        }
        sl::Resource exposure{};
        sl::ResourceTag exposure_tag{};
        if (in.exposure && exposure_image_handle != VkImage{}) {
            ClearExposure(cmdbuf, exposure_image_handle, *in.exposure);
            exposure = Resource(exposure_image_handle, exposure_view_handle, EXPOSURE_FORMAT,
                                VK_IMAGE_LAYOUT_GENERAL, VkExtent2D{1, 1}, EXPOSURE_USAGE);
            static const sl::Extent exposure_area{0, 0, 1, 1};
            exposure_tag = sl::ResourceTag{&exposure, sl::kBufferTypeExposure,
                                           sl::ResourceLifecycle::eValidUntilEvaluate,
                                           &exposure_area};
            inputs[input_count++] = &exposure_tag;
        }
        const std::array to_general{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        const VkMemoryBarrier output_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        sl::Result api_result = sl::Result::eOk;
        const auto result = VideoCommon::RecordDlssEvaluation(
            [&] {
                if (!options_changed) {
                    return true;
                }
                const auto set = reinterpret_cast<PFun_slDLSSSetOptions*>(set_options);
                api_result = set(viewport, options);
                return api_result == sl::Result::eOk;
            },
            [&] {
                api_result = api.get_new_frame_token(frame, &index);
                return api_result == sl::Result::eOk && frame;
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);
            },
            [&] {
                api_result = api.set_constants(constants, *frame, viewport);
                return api_result == sl::Result::eOk;
            },
            [&] {
                api_result = api.evaluate_feature(sl::kFeatureDLSS, *frame, inputs.data(), input_count,
                                                  reinterpret_cast<sl::CommandBuffer*>(*cmdbuf));
                return api_result == sl::Result::eOk;
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, output_written, {},
                                       to_general);
            });
        if (result != VideoCommon::DlssEvaluationResult::Succeeded && !failed.exchange(true)) {
            LOG_ERROR(Render_Vulkan,
                      "NVIDIA DLSS recording failed at stage {} (Streamline result {}); using FSR 3.1",
                      static_cast<int>(result), static_cast<int>(api_result));
        }
        if (result != VideoCommon::DlssEvaluationResult::Succeeded) {
            clear_output();
        }
    });
    // DLSS binds its own pipelines and descriptors on the command buffer.
    scheduler.InvalidateState();
    return true;
}

} // namespace Vulkan
