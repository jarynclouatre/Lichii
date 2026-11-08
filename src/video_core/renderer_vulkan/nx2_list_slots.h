// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/nx2_list_indirect.h"
#include "video_core/nx2_list_indirect_stats.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class MemoryAllocator;
class Scheduler;

/// The split GPU thread's back's slots for GPU-written command list words the host GPU takes
/// over (Nx2ListIndirect). A slot is SlotBytes of device-local memory: the list's words as the
/// back found them at the list's place (the snapshot, raw), and at DispatchOffset the same three
/// as DispatchNativeCompute takes a grid: width, height & 0xffff, depth & 0xffff (a
/// VkDispatchIndirectCommand). Holds keep a slot from being used again: the front's while its
/// macros' words hold the slot's placeholders, the back's while a QMD's grid or a word compute
/// dispatches bind as constant buffer data holds them.
///
/// The words the front's inline uploads put placeholders in are owed words: while one holds its
/// placeholder in guest memory, a compute dispatch that binds it as constant buffer data gets the
/// slot word there (PatchUniform, recorded after the binding filled its buffer).
class Nx2ListSlots {
public:
    static constexpr u32 SlotCount = Tegra::Nx2ListIndirect::SlotCount;
    static constexpr u32 SlotBytes = 32;
    static constexpr u32 DispatchOffset = 16;

    explicit Nx2ListSlots(MemoryAllocator& memory_allocator_, Scheduler& scheduler_)
        : memory_allocator{memory_allocator_}, scheduler{scheduler_} {}

    /// Any thread (the front's): a slot nobody holds, held for the caller.
    [[nodiscard]] std::optional<u32> Acquire() {
        for (u32 i = 0; i < SlotCount; ++i) {
            const u32 slot = (next + i) % SlotCount;
            u32 expected = 0;
            if (holds[slot].compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                next = slot + 1;
                return slot;
            }
        }
        return std::nullopt;
    }

    /// Any thread: drops a hold of `slot`.
    void Release(u32 slot);

    [[nodiscard]] bool Held(u32 slot) const {
        return slot < SlotCount && holds[slot].load(std::memory_order_acquire) != 0;
    }

    /// The back: records the copy of the `count` words at `src_offset` of `src` (the list's host
    /// buffer, the copy's use of the words marked) into `slot`.
    void Snapshot(VkBuffer src, u64 src_offset, u32 count, u32 slot);

    /// The back: the word at `device_addr` was just written `value` (a placeholder) by an inline
    /// upload; `guest_word` is what guest memory holds there now.
    void NoteOwed(DAddr device_addr, u32 value, u32 guest_word);

    /// The back: an inline QMD with grid words x, y, z was just written at `device_addr`.
    void NoteInlineQmdGrid(DAddr device_addr, u32 x, u32 y, u32 z);

    [[nodiscard]] bool HasOwed() const {
        return !owed.empty();
    }

    /// The back, binding a compute dispatch's constant buffer [device_addr, +size) at `offset`
    /// of `dst`, filled already: records copies of the slot words owed there over it. `read`
    /// gives the word guest memory holds at a device address: a word that no longer holds its
    /// placeholder is no longer owed.
    template <typename Read>
    void PatchUniform(DAddr device_addr, u32 size, VkBuffer dst, u64 offset, Read&& read) {
        if (dst == VK_NULL_HANDLE || size == 0) {
            return;
        }
        copies.clear();
        patch_words.clear(); // TEST
        const DAddr end = device_addr + size;
        for (auto it = owed.lower_bound(device_addr >= 3 ? device_addr - 3 : 0);
             it != owed.end() && it->first < end;) {
            const DAddr word = it->first;
            const u32 value = it->second;
            if (read(word) != value) {
                Tegra::Nx2ListIndirect::Stats::Count(
                    Tegra::Nx2ListIndirect::Stats::Get().owed_dropped); // TEST
                Release(Tegra::Nx2ListIndirect::PlaceholderSlot(value));
                it = owed.erase(it);
                continue;
            }
            const DAddr first = (std::max)(word, device_addr);
            const DAddr last = (std::min)(word + 4, end);
            if (first < last) {
                const VkDeviceSize source =
                    VkDeviceSize{Tegra::Nx2ListIndirect::PlaceholderSlot(value)} * SlotBytes +
                    VkDeviceSize{Tegra::Nx2ListIndirect::PlaceholderWord(value)} * 4 +
                    (first - word);
                copies.push_back(
                    VkBufferCopy{source, offset + (first - device_addr), last - first});
                patch_words.push_back({value, static_cast<u32>(first - word)}); // TEST
            }
            ++it;
        }
        if (!copies.empty()) {
            ++patched_binds;
            Tegra::Nx2ListIndirect::Stats::Count(
                Tegra::Nx2ListIndirect::Stats::Get().patched_binds); // TEST
            RecordCopies(*buffer, dst, copies);
            if (Tegra::Nx2ListIndirect::Stats::g_check) [[unlikely]] {
                CheckPatch(dst);
            }
        }
    }

    /// The buffer and offset of `slot`'s VkDispatchIndirectCommand.
    [[nodiscard]] std::pair<VkBuffer, VkDeviceSize> DispatchArguments(u32 slot) const {
        return {*buffer, VkDeviceSize{slot} * SlotBytes + DispatchOffset};
    }

    /// The back: the raw words of `slots`, three each, once the host GPU wrote them (submits and
    /// waits for it).
    void Read(std::span<const u32> slots, std::span<u32> words);

    /// Dispatches whose constant buffers got owed words.
    [[nodiscard]] u64 PatchedBinds() const {
        return patched_binds;
    }

    /// TEST (EDEN_NX2_LIST_INDIRECT_CHECK): the words the front read as before for the list
    /// whose snapshot `slot` just got. What the host GPU then uses of the slot is compared with
    /// them once it ran (CheckFlush): the snapshot, the dispatches from it, the patches.
    void CheckSnapshot(u32 slot, std::span<const u32> words);
    /// TEST: the dispatch from `slot` just recorded.
    void CheckDispatch(u32 slot);

private:
    static constexpr u32 NoEntry = ~0U;
    static constexpr u32 SlotWordBytes = Tegra::Nx2ListIndirect::SlotWords * sizeof(u32);
    static constexpr u32 ReadbackSlots = 256;

    /// The slots' buffer, created on first use, zero-filled.
    void Prepare();

    /// Records `regions` from `src` to `dst` outside a render pass, between barriers as
    /// BufferCacheRuntime::CopyBuffer puts them: after every write and read before (a slot a copy
    /// writes may be read by an earlier dispatch or copy), before every read and write after.
    void RecordCopies(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> regions);

    /// Records a barrier that makes transfer writes visible to the host, after a fence wait.
    void RecordHostBarrier();

    // TEST (EDEN_NX2_LIST_INDIRECT_CHECK): copies of what the host GPU used, compared with what
    // the old path would have used once it ran them.
    struct CheckEntry {
        Tegra::Nx2ListIndirect::Stats::Check check;
        u32 slot;
        u32 bytes;
        std::array<u8, 12> expected;
    };
    struct PatchWord {
        u32 value; ///< The owed placeholder
        u32 skip;  ///< Bytes of it before the binding
    };
    static constexpr u32 CheckEntries = 1024;
    static constexpr u32 CheckEntryBytes = 16;
    /// Records a copy of the `expected.size()` bytes at `src_offset` of `src`, for CheckFlush.
    void CheckRecord(Tegra::Nx2ListIndirect::Stats::Check check, u32 slot, VkBuffer src,
                     VkDeviceSize src_offset, std::span<const u8> expected);
    /// The patch just recorded into `dst` (copies, patch_words).
    void CheckPatch(VkBuffer dst);
    /// Waits for the host GPU and compares what CheckRecord copied.
    void CheckFlush();

    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;
    vk::Buffer buffer;
    vk::Buffer readback; ///< Read's, host-visible
    std::array<std::atomic<u32>, SlotCount> holds{};
    u32 next{};                                              ///< Where Acquire looks first
    std::map<DAddr, u32> owed;                               ///< By device address (the back's)
    std::unordered_map<DAddr, std::array<u32, 3>> qmd_slots; ///< By QMD device address
    std::vector<VkBufferCopy> copies;
    u64 patched_binds{};
    // TEST (EDEN_NX2_LIST_INDIRECT_CHECK).
    std::vector<PatchWord> patch_words;
    vk::Buffer check_buffer;
    std::vector<CheckEntry> check_entries;
    std::array<std::array<u32, 3>, SlotCount> check_words{};
    std::array<u8, SlotCount> check_counts{}; ///< Words of check_words known (0: none)
};

} // namespace Vulkan
