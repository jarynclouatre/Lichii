// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA Streamline) in place of a game's DLSS
// evaluation. NVIDIA's DLLs come from the user's PC; see vulkan_common/streamline.h.

#pragma once

#include <array>
#include <atomic>
#include <optional>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

class StreamlineDlssUpscaler {
public:
    struct Image {
        VkImageView view;
        VkImage image;
        VkFormat format;
        VkImageSubresourceRange range;     ///< The view's level and layer, with the view's aspect.
        VkImageAspectFlags barrier_aspect; ///< Every aspect of the image's format.
    };

    struct Inputs {
        Image color;          ///< Jittered linear colour at the render size.
        Image depth;          ///< Device depth (reversed) at the render size.
        Image motion_vectors; ///< Motion to the previous frame at the render size.
        /// R16F mask at the render size (GENERAL layout): where DLSS should trust this frame's
        /// colour over its history (things without motion vectors). Optional.
        std::optional<Image> bias_mask;
        VkExtent2D render_size;
        VkExtent2D upscale_size;
        /// This frame's sample offset in render pixels, as the game hands its own DLSS.
        std::array<f32, 2> jitter;
        /// Turns a motion vector into render pixels (the render size for UV offsets).
        std::array<f32, 2> motion_vector_scale;
        bool reset;
        /// Depth is reversed (1 near, 0 far); Ocarina of Time's is not.
        bool depth_inverted = true;
        /// -1: EDEN_NX2_DLSS_HDR decides; 0: colour is tonemapped LDR; 1: linear HDR.
        int color_hdr = -1;
        /// sl::DLSSPreset for this caller when EDEN_NX2_DLSS_PRESET is unset; -1: preset K.
        int preset = -1;
        /// The game's exposure for HDR colour (what its tonemap multiplies the colour by), given
        /// to DLSS in place of its auto exposure. Optional.
        std::optional<f32> exposure;
        /// The game's camera (OpenGL clip space, row-major, column vectors), for the constants
        /// frame generation reads; without it they are identities.
        struct Camera {
            /// Previous clip from current NDC (z = depth * 2 - 1).
            std::array<f32, 16> reproject;
            std::array<f32, 16> view_projection;
            /// The image's first row is NDC y = +1.
            bool flip_y;
        };
        std::optional<Camera> camera;
    };

    /// Throws vk::Exception when Streamline has no DLSS for this device.
    explicit StreamlineDlssUpscaler(const Device& device, Scheduler& scheduler,
                                    MemoryAllocator& memory_allocator);
    ~StreamlineDlssUpscaler();

    /// Records one evaluation into the output image (RGBA16F at the upscale size, GENERAL
    /// layout). Returns false when DLSS can't run, so the caller can fall back to FSR 3.1.
    bool Upscale(const Inputs& inputs);


    [[nodiscard]] VkImage OutputImage() const noexcept {
        return *output_image;
    }
    [[nodiscard]] VkImageView OutputView() const noexcept {
        return *output_view;
    }
    /// Nearest-neighbour clamp sampler (normalized coordinates), for reading the output.
    [[nodiscard]] VkSampler PointSampler() const noexcept {
        return *point_sampler;
    }

private:
    void CreateOutput(VkExtent2D upscale_size);
    void CreateExposure();
    static void ClearExposure(vk::CommandBuffer cmdbuf, VkImage image, f32 value);

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& memory_allocator;
    void* set_options{};     ///< slDLSSSetOptions
    u32 applied_preset_value{~0u}; ///< sl::DLSSPreset last given to DLSS
    VkExtent2D render_extent{};
    VkExtent2D upscale_extent{};
    u32 frame_index{};
    vk::Image output_image;
    vk::ImageView output_view;
    vk::Image exposure_image; ///< 1x1 R32F, cleared to Inputs::exposure before each evaluation
    vk::ImageView exposure_view;
    bool applied_auto_exposure{};
    vk::Sampler point_sampler;
    std::atomic<bool> failed{};
};

} // namespace Vulkan
