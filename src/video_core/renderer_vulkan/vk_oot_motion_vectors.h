// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Motion vectors (R16G16F, UV offsets to the previous frame) for Ocarina of Time's DLSS mode
/// (host_shaders/oot_motion_vectors.comp): camera motion from depth and a reprojection matrix,
/// replaced by the object motion image (vk_oot_object_motion.h) where it holds the visible
/// fragment.
class OotMotionVectors {
public:
    explicit OotMotionVectors(const Device& device, MemoryAllocator& allocator);
    ~OotMotionVectors();

    /// Records the pass; returns the motion vector image (GENERAL layout) at `size`.
    /// `object_pixels` (two words per pixel of `size`, OotObjectMotion) may be null. With
    /// `effects_view` (any size), the mask image gets its brightest channel times `mask_gain`.
    VkImageView Generate(Scheduler& scheduler, VkImageView depth_view, VkExtent2D size,
                         const std::array<f32, 16>& reproject, std::array<f32, 2> jitter,
                         bool flip_y, VkBuffer object_pixels, VkImageView effects_view = {},
                         f32 mask_gain = 0.0f, VkImageView fog_view = {}, f32 fog_gain = 0.0f,
                         VkImageView color_view = {});

    /// Copies an image (GENERAL layout) into copy `which` (0: the scene colour before the
    /// particles, 1: with them, 2: the tonemapped image with them), or with `restore` back.
    void SnapshotScene(Scheduler& scheduler, VkImage image, VkFormat format, VkExtent2D size,
                       u32 which, bool restore = false);

    /// Adds what the particles added to the tonemapped image (copy 2 minus `without_view`, both
    /// at the render size) to DLSS's output (RGBA16F storage image, GENERAL layout).
    bool CompositeParticles(Scheduler& scheduler, VkImageView output_view, VkExtent2D output_size,
                            VkImageView without_view, VkExtent2D render_size);

    [[nodiscard]] VkImage Image() const noexcept {
        return *image;
    }
    [[nodiscard]] VkImageView View() const noexcept {
        return *view;
    }
    /// DLSS's bias-to-current-colour mask (R16F, GENERAL layout), written with the motion.
    [[nodiscard]] VkImage MaskImage() const noexcept {
        return *mask_image;
    }
    [[nodiscard]] VkImageView MaskView() const noexcept {
        return *mask_view;
    }

private:
    static constexpr u32 SetCount = 4;

    void CreateImage(VkExtent2D size);
    void ReportStats();

    const Device& device;
    MemoryAllocator& allocator;
    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSetLayout descriptor_set_layout;
    vk::DescriptorSets descriptor_sets;
    std::array<u64, SetCount> set_ticks{};
    u32 next_set{};
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule shader;
    vk::Pipeline pipeline;
    vk::Sampler sampler;
    vk::Sampler linear_sampler;
    vk::Image image;
    vk::ImageView view;
    vk::Image mask_image;
    vk::ImageView mask_view;
    // EDEN_OOT_MOTION_CHECK: this frame's and last frame's colour, swapped every evaluation.
    std::array<vk::Image, 3> scene_images;
    std::array<vk::ImageView, 3> scene_views;
    std::array<VkExtent2D, 3> scene_extents{};
    std::array<VkFormat, 3> scene_formats{};
    vk::DescriptorPool composite_pool;
    vk::DescriptorSetLayout composite_set_layout;
    vk::DescriptorSets composite_sets;
    std::array<u64, SetCount> composite_ticks{};
    u32 next_composite_set{};
    vk::PipelineLayout composite_layout;
    vk::ShaderModule composite_shader;
    vk::Pipeline composite_pipeline;
    std::array<vk::Image, 2> history_images;
    std::array<vk::ImageView, 2> history_views;
    u32 history_index{};
    bool history_valid{};
    VkExtent2D extent{};
    bool image_initialized{};
    // Bound when there is no object motion.
    vk::Buffer dummy_pixels;
    // EDEN_OOT_OBJECT_MOTION_STATS=1: per-pixel counts of the object motion use.
    vk::Buffer stats;
    std::array<u32, 15> reported{};
    u32 generated{};
};

} // namespace Vulkan
