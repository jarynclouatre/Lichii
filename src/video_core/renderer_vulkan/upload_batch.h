// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// Uploads in order (Scheduler::RecordUpload) waiting to be recorded as one command: their
/// copies run between one barrier before and one after. Copies of one batch may run in any
/// order and at once, so a batch never holds two copies that write the same bytes, or one that
/// writes a buffer another reads.
class UploadBatch {
public:
    /// Copies with the same source and destination, consecutive in `copies`.
    struct Group {
        VkBuffer src;
        VkBuffer dst;
        u32 first;
        u32 count;
    };

    /// More copies than this go in a later batch. It bounds CanAdd's checks on the recording
    /// thread, at the cost of two barriers per this many uploads.
    static constexpr size_t MaxCopies = 64;

    [[nodiscard]] bool Empty() const noexcept {
        return copies.empty();
    }

    /// Whether these copies can join the batch: it has room, they read no buffer a copy in it
    /// writes and write none it reads, and they write no byte of dst that a copy in it writes.
    /// Otherwise the batch is recorded first.
    [[nodiscard]] bool CanAdd(VkBuffer src, VkBuffer dst,
                              std::span<const VkBufferCopy> new_copies) const {
        if (copies.size() + new_copies.size() > MaxCopies) {
            return false;
        }
        for (const Group& group : groups) {
            if (src == group.dst || dst == group.src) {
                return false;
            }
            if (group.dst != dst) {
                continue;
            }
            for (u32 index = group.first; index < group.first + group.count; ++index) {
                const VkBufferCopy& pending = copies[index];
                for (const VkBufferCopy& copy : new_copies) {
                    if (copy.dstOffset < pending.dstOffset + pending.size &&
                        pending.dstOffset < copy.dstOffset + copy.size) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /// Records a batch: the barrier before an in-order buffer copy (BufferCacheRuntime::
    /// CopyBuffer's), the copies, and the barrier after one, made to reach transfers as well.
    /// Recorded one by one, the barriers between two uploads ordered the copies before them
    /// ahead of the copies after them, which CanAdd makes needless, and ahead of all later work:
    /// graphics and compute through the barrier after a copy, transfers (in later submissions
    /// too) through the barrier before the next one. The barrier after the batch does both.
    static void Record(vk::CommandBuffer cmdbuf, std::span<const VkBufferCopy> batch_copies,
                       std::span<const Group> batch_groups) {
        static constexpr VkMemoryBarrier READ_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, READ_BARRIER);
        for (const Group& group : batch_groups) {
            cmdbuf.CopyBuffer(group.src, group.dst,
                              vk::Span<VkBufferCopy>(batch_copies.data() + group.first,
                                                     group.count));
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0, WRITE_BARRIER);
    }

    void Add(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> new_copies) {
        const u32 first = static_cast<u32>(copies.size());
        const u32 count = static_cast<u32>(new_copies.size());
        copies.insert(copies.end(), new_copies.begin(), new_copies.end());
        if (!groups.empty() && groups.back().src == src && groups.back().dst == dst) {
            groups.back().count += count;
        } else {
            groups.push_back({src, dst, first, count});
        }
    }

    std::vector<VkBufferCopy> copies;
    std::vector<Group> groups;
};

/// Uploads that run in the upload command buffer, ahead of all the work of their submission's
/// main command buffer (Scheduler::RecordReorderedUpload). They are recorded as one command when
/// the submission is: a copy per source and destination. Nothing orders them against each other
/// and nothing has to: the buffer cache keeps an upload in order when work recorded before it in
/// the submission, an earlier upload included, uses a byte it writes (DecideUploadOrder).
class ReorderedUploadBatch {
public:
    [[nodiscard]] bool Empty() const noexcept {
        return entries.empty();
    }

    /// prior_tick: the newest earlier submission whose work uses bytes the copies write, or a
    /// later one; 0 if none.
    void Add(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> new_copies, u64 prior_tick) {
        for (const VkBufferCopy& copy : new_copies) {
            entries.push_back(Entry{src, dst, copy});
        }
        newest_prior_tick = (std::max)(newest_prior_tick, prior_tick);
    }

    /// Empties the batch into its copies, grouped by source and destination (in the order they
    /// were added within a group), and the newest earlier submission they must come after.
    void Take(std::vector<VkBufferCopy>& copies, std::vector<UploadBatch::Group>& groups,
              u64& prior_tick) {
        std::stable_sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
            if (lhs.src != rhs.src) {
                return std::less<VkBuffer>{}(lhs.src, rhs.src);
            }
            return std::less<VkBuffer>{}(lhs.dst, rhs.dst);
        });
        copies.clear();
        groups.clear();
        for (const Entry& entry : entries) {
            const u32 index = static_cast<u32>(copies.size());
            copies.push_back(entry.copy);
            if (!groups.empty() && groups.back().src == entry.src &&
                groups.back().dst == entry.dst) {
                ++groups.back().count;
            } else {
                groups.push_back(UploadBatch::Group{entry.src, entry.dst, index, 1});
            }
        }
        prior_tick = newest_prior_tick;
        entries.clear();
        newest_prior_tick = 0;
    }

    /// Records a batch into the upload command buffer. With after_prior_work it begins with a
    /// barrier whose first scope is all the work submitted before it, earlier submissions' main
    /// command buffers included: their reads and writes of the bytes the copies write come first.
    /// The upload command buffer's last barrier (Scheduler::SubmitExecution) orders the copies
    /// before the main command buffer's work and all later work.
    static void Record(vk::CommandBuffer upload_cmdbuf, std::span<const VkBufferCopy> batch_copies,
                       std::span<const UploadBatch::Group> batch_groups, bool after_prior_work) {
        static constexpr VkMemoryBarrier PRIOR_WORK_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        if (after_prior_work) {
            upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, PRIOR_WORK_BARRIER);
        }
        for (const UploadBatch::Group& group : batch_groups) {
            upload_cmdbuf.CopyBuffer(
                group.src, group.dst,
                vk::Span<VkBufferCopy>(batch_copies.data() + group.first, group.count));
        }
    }

private:
    struct Entry {
        VkBuffer src;
        VkBuffer dst;
        VkBufferCopy copy;
    };

    std::vector<Entry> entries;
    u64 newest_prior_tick = 0;
};

} // namespace Vulkan
