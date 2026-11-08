// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

/// Where an upload into a buffer runs.
struct UploadOrder {
    /// In the submission's upload command buffer, ahead of all the work of its main one. If not,
    /// at its place in the main command buffer, between barriers, after the work recorded before
    /// it (a render pass in progress ends first).
    bool reorder{};
    /// When reordered: the newest earlier submission whose work used the buffer, 0 if none. The
    /// upload command buffer begins with a barrier on earlier work unless the host GPU finished
    /// that submission by the time the upload command buffer is recorded.
    u64 prior_tick{};
};

/// The bytes of a buffer that GPU work recorded in a submission uses (reads or writes), by the
/// tick of the submission. Byte ranges are kept for the newest submission only: the ones before
/// it collapse into the tick of the newest of them, which no byte's last use is later than.
class SubmissionUses {
public:
    /// Disjoint ranges are merged with their neighbors beyond this many: the uses then cover
    /// more bytes than the work used, never fewer.
    static constexpr size_t MaxRanges = 1024;

    /// Work recorded in submission `tick` uses [offset, offset + size). Ticks never go back.
    void Track(u64 offset, u64 size, u64 tick) noexcept {
        if (size == 0) {
            return;
        }
        if (tick != current_tick) {
            if (tick < current_tick) {
                previous_tick = (std::max)(previous_tick, tick);
                return;
            }
            if (!ranges.empty()) {
                previous_tick = current_tick;
            }
            ranges.clear();
            recent.fill(Range{});
            current_tick = tick;
        }
        const u64 begin = offset;
        const u64 end = offset + size;
        // Draws bind the same ranges again and again. Ranges only grow until the tick changes,
        // so one that covered these bytes still does.
        for (const Range& range : recent) {
            if (range.begin <= begin && end <= range.end) {
                return;
            }
        }
        recent[recent_cursor++ % recent.size()] = Range{begin, end};
        // The first range that ends at or after begin: it overlaps or touches [begin, end), or
        // lies after it.
        auto first =
            std::lower_bound(ranges.begin(), ranges.end(), begin,
                             [](const Range& range, u64 value) { return range.end < value; });
        if (first != ranges.end() && first->begin <= begin && end <= first->end) {
            return;
        }
        u64 merged_begin = begin;
        u64 merged_end = end;
        auto last = first;
        while (last != ranges.end() && last->begin <= end) {
            merged_begin = (std::min)(merged_begin, last->begin);
            merged_end = (std::max)(merged_end, last->end);
            ++last;
        }
        if (first == last) {
            ranges.insert(first, Range{merged_begin, merged_end});
        } else {
            *first = Range{merged_begin, merged_end};
            ranges.erase(first + 1, last);
        }
        if (ranges.size() > MaxRanges) {
            Coarsen();
        }
    }

    /// Whether work recorded in submission `tick` uses a byte of [offset, offset + size).
    [[nodiscard]] bool IsUsed(u64 offset, u64 size, u64 tick) const noexcept {
        if (size == 0 || tick != current_tick || ranges.empty()) {
            return false;
        }
        const u64 begin = offset;
        const u64 end = offset + size;
        // The first range that ends after begin.
        const auto it =
            std::upper_bound(ranges.begin(), ranges.end(), begin,
                             [](u64 value, const Range& range) { return value < range.end; });
        return it != ranges.end() && it->begin < end;
    }

    /// The newest submission before `tick` whose work used a byte of the buffer, 0 if none did.
    [[nodiscard]] u64 NewestTickBefore(u64 tick) const noexcept {
        // current_tick is 0 until the first use. Asked about the past (current_tick > tick), the
        // newest use known is still a bound.
        return current_tick == tick ? previous_tick : current_tick;
    }

    /// How many disjoint ranges the newest submission's work uses (for tests).
    [[nodiscard]] size_t RangeCount() const noexcept {
        return ranges.size();
    }

private:
    struct Range {
        u64 begin;
        u64 end;
    };

    /// Merges each pair of neighbors, gap included.
    void Coarsen() noexcept {
        size_t out = 0;
        for (size_t in = 0; in < ranges.size(); in += 2) {
            const u64 merged_end = in + 1 < ranges.size() ? ranges[in + 1].end : ranges[in].end;
            ranges[out++] = Range{ranges[in].begin, merged_end};
        }
        ranges.resize(out);
        recent.fill(Range{});
    }

    /// Sorted, disjoint, and never touching: neighbors merge.
    std::vector<Range> ranges;
    /// Ranges tracked lately in current_tick, as given.
    std::array<Range, 4> recent{};
    u32 recent_cursor{};
    u64 current_tick{};
    u64 previous_tick{};
};

/// Where an upload that writes `copies` (dst_offset, size) of a buffer with these uses runs when
/// it is recorded in submission `tick`: in order when work recorded before it in the submission
/// uses a byte it writes, otherwise in the upload command buffer, after the work of the earlier
/// submissions that used the buffer. The caller then tracks the copies as uses: an upload of the
/// same bytes later in the submission stays in order behind this one.
template <typename Copies>
[[nodiscard]] UploadOrder DecideUploadOrder(const SubmissionUses& uses, const Copies& copies,
                                            u64 tick) {
    for (const auto& copy : copies) {
        if (uses.IsUsed(copy.dst_offset, copy.size, tick)) {
            return UploadOrder{};
        }
    }
    return UploadOrder{.reorder = true, .prior_tick = uses.NewestTickBefore(tick)};
}

} // namespace VideoCommon
