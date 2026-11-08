// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_upload_reorder_check.h"
#include "video_core/texture_cache/types.h"

namespace Vulkan {
namespace {

constexpr size_t NUM_SOURCES = static_cast<size_t>(UploadReorderCheck::Source::Count);
constexpr std::array<const char*, NUM_SOURCES> SOURCE_NAMES{
    "a marked use",         "a descriptor",
    "a texel buffer",       "a vertex buffer",
    "an index buffer",      "transform feedback",
    "a multi-range source", "a copy reading",
    "a copy writing",       "a clear",
};
static_assert(SOURCE_NAMES[NUM_SOURCES - 1] != nullptr);

struct Range {
    u64 end;
    u32 sources; ///< bit per Source
};

/// Byte ranges by their begin, disjoint. Overlapping ranges merge, their sources too; touching
/// ones merge when they have the same sources, so an error names what used the bytes.
using RangeSet = std::map<u64, Range>;

void Insert(RangeSet& set, u64 begin, u64 end, u32 sources) {
    auto it = set.upper_bound(begin);
    if (it != set.begin()) {
        const auto before = std::prev(it);
        if (before->second.end >= end && (before->second.sources & sources) == sources) {
            // Noted already (draws bind the same ranges again and again).
            return;
        }
        if (before->second.end > begin ||
            (before->second.end == begin && before->second.sources == sources)) {
            it = before;
        }
    }
    u64 merged_begin = begin;
    u64 merged_end = end;
    u32 merged_sources = sources;
    while (it != set.end() && (it->first < merged_end ||
                               (it->first == merged_end && it->second.sources == merged_sources))) {
        merged_begin = (std::min)(merged_begin, it->first);
        merged_end = (std::max)(merged_end, it->second.end);
        merged_sources |= it->second.sources;
        it = set.erase(it);
    }
    set.emplace(merged_begin, Range{merged_end, merged_sources});
}

/// The sources of the ranges that share a byte with [begin, end), 0 when none does.
u32 Overlap(const RangeSet& set, u64 begin, u64 end) {
    auto it = set.upper_bound(begin);
    if (it != set.begin()) {
        --it;
    }
    u32 sources = 0;
    for (; it != set.end() && it->first < end; ++it) {
        if (it->second.end > begin) {
            sources |= it->second.sources;
        }
    }
    return sources;
}

/// A handle's value for the log: a pointer on 64-bit hosts, an integer on 32-bit ones.
template <typename Handle>
u64 HandleValue(Handle handle) {
    if constexpr (std::is_pointer_v<Handle>) {
        return static_cast<u64>(reinterpret_cast<uintptr_t>(handle));
    } else {
        return static_cast<u64>(handle);
    }
}

std::string SourceList(u32 sources) {
    std::string list;
    for (size_t i = 0; i < NUM_SOURCES; ++i) {
        if ((sources >> i) & 1) {
            list += fmt::format("{}{}", list.empty() ? "" : ", ", SOURCE_NAMES[i]);
        }
    }
    return list;
}

struct Counts {
    u64 uploads;
    u64 reordered;
    u64 reordered_after_unfinished;
    u64 in_order;
    u64 in_order_used;
    u64 past_use;
    u64 unordered;
};

} // Anonymous namespace

struct UploadReorderCheck::Impl {
    explicit Impl(Scheduler& scheduler_) : scheduler{scheduler_} {}

    struct TickUses {
        u64 tick;
        RangeSet ranges;
    };
    struct BufferUses {
        /// Oldest first, one entry a tick.
        std::vector<TickUses> ticks;
    };

    /// Drops the uses of submissions the host GPU finished: nothing recorded now races them.
    void Prune(BufferUses& uses) const {
        const u64 current = scheduler.CurrentTick();
        auto it = uses.ticks.begin();
        while (it != uses.ticks.end() && it->tick < current && scheduler.IsFree(it->tick)) {
            ++it;
        }
        uses.ticks.erase(uses.ticks.begin(), it);
    }

    void Sweep() {
        for (auto it = buffers.begin(); it != buffers.end();) {
            Prune(it->second);
            it = it->second.ticks.empty() ? buffers.erase(it) : std::next(it);
        }
    }

    void Report(std::chrono::steady_clock::time_point now) {
        const double seconds = std::chrono::duration<double>(now - window_start).count();
        LOG_INFO(Render_Vulkan,
                 "Upload reorder check in {:.1f} s: {} uploads from the stream buffer: {} "
                 "reordered ({} of them after unfinished work of earlier submissions on their "
                 "bytes), {} in order ({} after a use of their bytes in the same submission); "
                 "errors: {} reordered past a use of their bytes, {} reordered ahead of unfinished "
                 "earlier work on their bytes without waiting for it (since start: {} and {} in {} "
                 "uploads)",
                 seconds, window.uploads, window.reordered, window.reordered_after_unfinished,
                 window.in_order, window.in_order_used, window.past_use, window.unordered,
                 total.past_use, total.unordered, total.uploads);
        window = {};
        window_start = now;
    }

    Scheduler& scheduler;
    std::mutex mutex;
    std::unordered_map<VkBuffer, BufferUses> buffers;
    u64 notes_since_sweep{};
    Counts window{};
    Counts total{};
    u64 errors_logged{};
    std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
};

std::unique_ptr<UploadReorderCheck> UploadReorderCheck::Create(Scheduler& scheduler) {
    const char* const value = std::getenv("EDEN_NX2_UPLOAD_REORDER_CHECK");
    if (value == nullptr || value[0] == '\0' || value[0] == '0') {
        return nullptr;
    }
    std::unique_ptr<UploadReorderCheck> check{new UploadReorderCheck(scheduler)};
    active = check.get();
    LOG_INFO(Render_Vulkan,
             "Upload reorder check (EDEN_NX2_UPLOAD_REORDER_CHECK): checking the place of every "
             "upload from the stream buffer against an exact model of the bytes each "
             "submission's work uses");
    return check;
}

UploadReorderCheck::UploadReorderCheck(Scheduler& scheduler)
    : impl{std::make_unique<Impl>(scheduler)} {}

UploadReorderCheck::~UploadReorderCheck() {
    if (active == this) {
        active = nullptr;
    }
}

void UploadReorderCheck::NoteUse(VkBuffer buffer, u64 offset, u64 size, Source source) {
    if (buffer == VK_NULL_HANDLE || size == 0) {
        return;
    }
    Impl& self = *impl;
    std::scoped_lock lock{self.mutex};
    const u64 tick = self.scheduler.CurrentTick();
    Impl::BufferUses& uses = self.buffers[buffer];
    self.Prune(uses);
    if (uses.ticks.empty() || uses.ticks.back().tick < tick) {
        uses.ticks.push_back(Impl::TickUses{tick, {}});
    }
    // A whole-size binding ends at the end of the address space instead of wrapping around.
    const u64 end = size > ~u64{0} - offset ? ~u64{0} : offset + size;
    Insert(uses.ticks.back().ranges, offset, end, 1u << static_cast<u32>(source));
    if (++self.notes_since_sweep >= 1u << 16) {
        self.notes_since_sweep = 0;
        self.Sweep();
    }
}

void UploadReorderCheck::CheckUpload(VkBuffer buffer,
                                     std::span<const VideoCommon::BufferCopy> copies,
                                     VideoCommon::UploadOrder order, std::source_location site) {
    if (copies.empty()) {
        return;
    }
    Impl& self = *impl;
    std::scoped_lock lock{self.mutex};
    const u64 tick = self.scheduler.CurrentTick();
    // What work recorded before this upload in its submission uses of its bytes, and the newest
    // earlier submission not known finished whose work does.
    u32 used_now = 0;
    u64 unfinished = 0;
    const VideoCommon::BufferCopy* used_copy = nullptr;
    if (const auto it = self.buffers.find(buffer); it != self.buffers.end()) {
        self.Prune(it->second);
        for (const Impl::TickUses& entry : it->second.ticks) {
            for (const VideoCommon::BufferCopy& copy : copies) {
                const u32 sources =
                    Overlap(entry.ranges, copy.dst_offset, copy.dst_offset + copy.size);
                if (sources == 0) {
                    continue;
                }
                if (entry.tick == tick) {
                    used_now |= sources;
                    used_copy = used_copy ? used_copy : &copy;
                } else if (entry.tick < tick && !self.scheduler.IsFree(entry.tick)) {
                    unfinished = (std::max)(unfinished, entry.tick);
                }
            }
        }
    }
    // A reordered upload waits for earlier work only through the submission's barrier, which is
    // left out once the host GPU has finished order.prior_tick: no unfinished work after that
    // submission may use its bytes.
    const bool reordered = order.reorder;
    const bool past_use = reordered && used_now != 0;
    const bool unordered = reordered && unfinished > order.prior_tick;
    for (Counts* const counts : {&self.window, &self.total}) {
        ++counts->uploads;
        if (reordered) {
            ++counts->reordered;
            counts->reordered_after_unfinished += unfinished != 0 ? 1 : 0;
        } else {
            ++counts->in_order;
            counts->in_order_used += used_now != 0 ? 1 : 0;
        }
        counts->past_use += past_use ? 1 : 0;
        counts->unordered += unordered ? 1 : 0;
    }
    if ((past_use || unordered) &&
        (self.errors_logged++ < 64 || (self.errors_logged & 1023) == 0)) {
        const VideoCommon::BufferCopy& copy = used_copy ? *used_copy : copies.front();
        if (past_use) {
            LOG_ERROR(Render_Vulkan,
                      "Upload reorder check: an upload into buffer {:#x} [{:#x}, {:#x}) at {}:{} "
                      "was put in the upload command buffer of submission {}, ahead of {} of its "
                      "bytes recorded before it in that submission",
                      HandleValue(buffer), copy.dst_offset, copy.dst_offset + copy.size,
                      site.file_name(), site.line(), tick, SourceList(used_now));
        } else {
            LOG_ERROR(Render_Vulkan,
                      "Upload reorder check: an upload into buffer {:#x} at {}:{} was put in the "
                      "upload command buffer of submission {} after submission {} only, while "
                      "work of submission {} using its bytes is not known finished",
                      HandleValue(buffer), site.file_name(), site.line(), tick, order.prior_tick,
                      unfinished);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - self.window_start >= std::chrono::seconds{10}) {
        self.Report(now);
    }
}

} // namespace Vulkan
