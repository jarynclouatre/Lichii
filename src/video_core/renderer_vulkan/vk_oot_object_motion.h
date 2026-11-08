// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <unordered_map>

#include "common/common_types.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Resources of Ocarina of Time's per-object motion vectors (docs/nx2/oot-motion-vectors-plan.md):
/// a history buffer of clip positions, two halves swapped every frame, with a slot range per draw
/// identity, and the per-pixel buffer where the scene's fragment shaders keep the nearest
/// fragment's motion and depth (u64: depth bits << 32 | half2 motion; all ones means none).
class OotObjectMotion {
public:
    using Layout = Shader::Backend::SPIRV::OotMotionLayout;

    explicit OotObjectMotion(const Device& device, MemoryAllocator& allocator);
    ~OotObjectMotion();

    /// The instance graphics pipelines bind (created by the rasterizer), or null.
    [[nodiscard]] static OotObjectMotion* Get() noexcept;

    /// Sizes the pixel buffer for `size` (the scene's scaled size); records its first clear.
    void EnsureImage(Scheduler& scheduler, VkExtent2D size);

    /// Slots for one draw: `range` vertices per instance after subtracting the biases.
    /// `by_order`: a draw repeated within a frame takes the history of last frame's draw at the
    /// same position in the order (else none).
    [[nodiscard]] Layout Allocate(u64 identity, u32 range, u32 instances, u32 vertex_bias,
                                  u32 instance_bias, bool by_order = false);

    /// Called once per presented frame: swaps the history halves and orders this frame's history
    /// writes before the next frame's reads.
    void BeginFrame(Scheduler& scheduler);

    /// The combine pass read and cleared the pixel buffer this frame.
    void NoteCombined() noexcept {
        combined = true;
    }

    [[nodiscard]] VkBuffer HistoryBuffer() const noexcept {
        return *history;
    }
    [[nodiscard]] VkDeviceAddress HistoryAddress() const noexcept {
        return history_address;
    }
    [[nodiscard]] VkDeviceSize HistorySize() const noexcept {
        return HistoryBytes;
    }
    [[nodiscard]] VkBuffer PixelBuffer() const noexcept {
        return *pixels;
    }
    [[nodiscard]] VkDeviceAddress PixelAddress() const noexcept {
        return pixels_address;
    }
    [[nodiscard]] VkDeviceSize PixelBytes() const noexcept {
        return VkDeviceSize{extent.width} * extent.height * 8;
    }
    [[nodiscard]] VkExtent2D Extent() const noexcept {
        return extent;
    }
    [[nodiscard]] bool HasImage() const noexcept {
        return static_cast<bool>(pixels);
    }

private:
    static constexpr u32 SlotsPerHalf = 4u << 20;
    static constexpr VkDeviceSize HistoryBytes = VkDeviceSize{SlotsPerHalf} * 2 * 16;

    struct Slot {
        u32 base;
        u32 range;
        u32 instances;
    };

    void ClearPixels(Scheduler& scheduler);

    const Device& device;
    MemoryAllocator& allocator;
    vk::Buffer history;
    VkDeviceAddress history_address{};
    bool history_initialized{};
    vk::Buffer pixels;
    VkDeviceAddress pixels_address{};
    VkExtent2D extent{};

    std::unordered_map<u64, Slot> current;
    std::unordered_map<u64, Slot> previous;
    std::unordered_map<u64, u32> occurrences;
    std::unordered_map<u64, u32> previous_occurrences;
    u32 half{};
    u32 used{};
    bool combined{};
    bool drawn{};

    struct Stats {
        u64 frames{};
        u64 draws{};
        u64 matched{};
        u64 overflow{};
        u64 repeated{};
        u64 slots{};
    } stats;
};

} // namespace Vulkan
