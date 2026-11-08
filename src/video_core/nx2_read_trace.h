// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/gpu_wait_stats.h"

// TEST (EDEN_NX2_READ_TRACE=1): where a guest read of GPU-written memory spends its wait.
//
// Each read that waits (GPU::OnCPURead) is placed on the host GPU's timeline. Vulkan writes a
// device timestamp at the start and end of every submission, after each draw or dispatch that
// writes pages the guest reads back (the area's last such write before the read's copy is "the
// write"), and before and after the read's copy. The reading thread reads the device's clock
// (VK_EXT_calibrated_timestamps) once the flush is asked for, which places the request on that
// timeline. The wait then splits into: until the write's work ended on the GPU (the GPU still
// had it, or work queued before it, to run), until the copy could start (work queued after the
// write), the copy, and the reading thread's wake-up, lock and copy into guest memory. Also kept:
// where the write was at the request (done, on the GPU, not yet queued) and how long before it
// the GPU thread recorded it, the work recorded between the write and the copy, the submissions
// in flight at the request with the GPU time they still ran, and how long the copy waited for
// its vkQueueSubmit. In the frames EDEN_DKB_FRAME_TRACE samples (each draw and dispatch timed),
// the GPU time before the copy also splits into timed draws and dispatches and the rest.
//
// Every 10 s: one "Read trace" line for all reads, and one for each of the areas that waited
// longest. Times are per read, in ms.
namespace VideoCommon::ReadTrace {

[[nodiscard]] inline bool SwitchSet() {
    const char* value = std::getenv("EDEN_NX2_READ_TRACE");
    return value != nullptr && *value != '\0' && *value != '0';
}

// A plain flag: checked at every read download and fence.
inline bool g_enabled = SwitchSet();

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_enabled = SwitchSet();
}

[[nodiscard]] inline bool Enabled() {
    return g_enabled;
}

/// Nanoseconds, the clock of the GPU wait stats (the same on every thread).
[[nodiscard]] inline u64 Now() {
    return GpuWaitStats::Now();
}

/// Work the GPU thread recorded so far.
struct Counts {
    u64 draws{};
    u64 dispatches{};
    u64 render_passes{};
};

/// The area's last GPU write before the read's copy was recorded.
struct Write {
    u64 tick{};     ///< The submission it went in
    u64 recorded{}; ///< Now() once the writing draw or dispatch was recorded
    u64 mark{};     ///< Device timestamp after it (Vulkan::Scheduler::TraceMark), 0 if none
    Counts counts{};
};

/// The copy a read waited for (BufferCache::DownloadMemoryForRead).
struct Copy {
    u64 tick{};     ///< Its submission
    u64 flushed{};  ///< Now() once the GPU thread had handed it over for submission
    u64 start{};    ///< Device timestamp before the copies: all earlier work done
    u64 end{};      ///< After them
    u64 sequence{}; ///< Writes the buffer cache noted before it have a lower one
    Counts counts{};
    bool early{}; ///< Recorded right behind its write (BufferCache::RecordEarlyDownloads)
};

enum class Step : u8 {
    Request,   ///< As the reading thread asks for the flush
    Calibrate, ///< While it waits for the GPU thread to run it: the device's clock
    Done,      ///< The bytes are in guest memory
};

/// One guest read that waited for the GPU, filled in as it goes.
struct Read {
    u64 area{};
    bool in_flight{}; ///< Waited for an earlier read's copy
    // Step::Request
    u64 request{};
    u64 completed{}; ///< The last submission the host GPU completed
    u64 submitted{}; ///< The last one vkQueueSubmit returned for
    u64 recording{}; ///< The one being recorded: those before it went to the Vulkan worker
    // Step::Calibrate: the device's timestamp `device` at Now() `host` (+- `uncertainty`)
    u64 device{};
    u64 host{};
    u64 uncertainty{};
    // FinishReadDownloads (the reading thread)
    bool has_copy{};
    Copy copy{};
    bool has_write{};
    Write write{};
    u64 woke{};   ///< The copy's submission was complete
    u64 locked{}; ///< Then the buffer cache lock was taken
    u64 landed{}; ///< Then the bytes were in guest memory
};

/// The reading thread's read while it is in RasterizerInterface::FinishRead.
inline thread_local Read* t_read{};

/// Where a read's wait went on the device's timeline (Vulkan::Scheduler, Step::Done).
struct Timeline {
    bool placed{}; ///< Calibrated, and the timestamps it needs were read
    /// ns from the request: until the write ended, until the copy could start, the copy, until
    /// the reading thread woke. They add up to `woke - request`.
    u64 to_write{};
    u64 to_copy{};
    u64 copy{};
    u64 to_wake{};
    /// Of to_write and to_copy, the time the host GPU ran no submission
    u64 idle_to_write{};
    u64 idle_to_copy{};
    /// GPU time the submissions in flight at the request still ran after it
    u64 backlog_busy{};
    /// The write's work ended this long before the request (0: after it)
    u64 write_lead{};
    /// From the write's end to the copy's start on the GPU, wherever the request fell
    u64 write_gap{};
    /// The copy's vkQueueSubmit returned this long after the GPU thread handed it over
    bool has_submit{};
    u64 submit{};
    /// Device ticks: the request, the write's end (not before the request), the copy's start
    std::array<u64, 3> window{};
};

/// The device timestamps placing a read takes (Vulkan::Scheduler).
struct DeviceTimes {
    double ns_per_tick{};
    u64 write_end{};
    u64 copy_start{};
    u64 copy_end{};
    /// Start and end of each submission after the last one the host GPU had completed at the
    /// request (read.completed + 1 on), up to the copy's.
    std::span<const std::pair<u64, u64>> submissions;
};

/// Splits a read's wait on the device's timeline.
[[nodiscard]] inline Timeline Place(const Read& read, const DeviceTimes& times) {
    // ns from the request: the device read `device` at Now() `host`.
    const auto at = [&](u64 ticks) {
        return static_cast<double>(static_cast<s64>(ticks - read.device)) * times.ns_per_tick +
               static_cast<double>(read.host - read.request);
    };
    // The host GPU's time running submissions in [lower, upper), those up to last_tick.
    const auto busy = [&](double lower, double upper, u64 last_tick) {
        double sum = 0.0;
        double covered = lower;
        for (size_t i = 0; i < times.submissions.size(); ++i) {
            if (read.completed + 1 + i > last_tick) {
                break;
            }
            const double from = (std::max)(at(times.submissions[i].first), covered);
            const double to = (std::min)(at(times.submissions[i].second), upper);
            if (to > from) {
                sum += to - from;
                covered = to;
            }
        }
        return sum;
    };
    const auto ns = [](double value) { return static_cast<u64>((std::max)(value, 0.0)); };
    const double woke = static_cast<double>(read.woke - read.request);
    const double write_ended = at(times.write_end);
    const double written = std::clamp(write_ended, 0.0, woke);
    const double copy_started = std::clamp(at(times.copy_start), written, woke);
    const double copied = std::clamp(at(times.copy_end), copy_started, woke);
    Timeline timeline{};
    timeline.placed = true;
    timeline.to_write = ns(written);
    timeline.to_copy = ns(copy_started - written);
    timeline.copy = ns(copied - copy_started);
    timeline.to_wake = ns(woke - copied);
    timeline.idle_to_write = ns(written - busy(0.0, written, read.copy.tick));
    timeline.idle_to_copy =
        ns(copy_started - written - busy(written, copy_started, read.copy.tick));
    timeline.backlog_busy = ns(busy(0.0, std::numeric_limits<double>::infinity(), read.submitted));
    timeline.write_lead = ns(-write_ended);
    timeline.write_gap = ns(at(times.copy_start) - write_ended);
    const double request_ticks = static_cast<double>(read.device) -
                                 static_cast<double>(read.host - read.request) / times.ns_per_tick;
    const auto ticks_at = [&](double offset) {
        return static_cast<u64>(request_ticks + offset / times.ns_per_tick);
    };
    timeline.window = {ticks_at(0.0), ticks_at(written), ticks_at(copy_started)};
    return timeline;
}

/// Early read downloads (BufferCache::RecordEarlyDownloads), any thread.
struct EarlyTotals {
    std::atomic<u64> recorded{};
    std::atomic<u64> bytes{};
    std::atomic<u64> served_landed{};    ///< Reads that found their bytes in guest memory
    std::atomic<u64> served_at_read{};   ///< Done on the host GPU: the read's lookup landed them
    std::atomic<u64> served_in_flight{}; ///< Not yet done: the read waited for them
    std::atomic<u64> gpu_side_waits{};   ///< Writes from the GPU's side that waited for one
    std::atomic<u64> gpu_side_wait_ns{};
};
inline EarlyTotals g_early;

inline void Count(std::atomic<u64>& counter, u64 amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

struct Totals {
    u64 reads{};
    u64 in_flight{};
    u64 early_copies{}; ///< The copy waited for was an early download
    u64 no_copy{};      ///< Found no copy to wait for
    u64 no_write{};     ///< No GPU write over the area seen since it was first read back
    u64 copy_done{};    ///< The copy's submission was complete at the request
    u64 wait{};         ///< Request until the bytes were in guest memory, all reads
    u64 lock{};         ///< Reads with a copy: woke until locked
    u64 land{};         ///< Locked until in guest memory
    // Placed reads
    u64 placed{};
    u64 to_write{};
    u64 to_copy{};
    u64 copy{};
    u64 to_wake{};
    u64 idle_to_write{};
    u64 idle_to_copy{};
    u64 backlog_busy{};
    u64 uncertainty{};
    u64 write_ended{}; ///< The write's work had ended at the request (write_lead is over these)
    u64 write_lead{};
    u64 write_gap{};
    // Reads with a copy: the submissions in flight, and those not yet queued, at the request
    u64 backlog_subs{};
    u64 unqueued_subs{};
    u64 submits{};
    u64 submit{};
    // Reads with a write: where it was at the request, when it was recorded, what came after it
    u64 write_done{};
    u64 write_on_gpu{};
    u64 write_unqueued{};  ///< Flushed to the Vulkan worker, no vkQueueSubmit yet
    u64 write_recording{}; ///< In the submission still being recorded
    u64 written_before{};  ///< Recorded before the request (since_write is over these)
    u64 since_write{};
    u64 draws{};
    u64 dispatches{};
    u64 render_passes{};
    u64 submissions{};
    // Placed reads in the frames EDEN_DKB_FRAME_TRACE samples
    u64 sampled{};
    u64 timed_to_write{};
    u64 busy_to_write{};
    u64 timed_to_copy{};
    u64 busy_to_copy{};
};

/// A placed read, until its frame's timed work arrives (NoteTimedWork).
struct Window {
    u64 area{};
    std::array<u64, 3> ticks{};
    u64 busy_to_write{};
    u64 busy_to_copy{};
    bool pending{};
};

struct State {
    std::mutex mutex;
    u64 window_start{};
    u64 presents{};
    Totals all;
    std::vector<std::pair<u64, Totals>> areas;
    Totals other_areas; ///< Past MaxAreas
    std::array<Window, 64> windows{};
    size_t next_window{};
};
inline State g_state;
inline constexpr size_t MaxAreas = 64;
inline constexpr size_t ShownAreas = 4;

/// `mutex` held.
inline Totals& AreaTotals(u64 area) {
    for (auto& [start, totals] : g_state.areas) {
        if (start == area) {
            return totals;
        }
    }
    if (g_state.areas.size() >= MaxAreas) {
        return g_state.other_areas;
    }
    return g_state.areas.emplace_back(area, Totals{}).second;
}

inline void Add(Totals& totals, const Read& read, const Timeline& timeline) {
    ++totals.reads;
    totals.in_flight += read.in_flight ? 1 : 0;
    totals.wait += read.landed - read.request;
    if (!read.has_copy) {
        ++totals.no_copy;
        return;
    }
    totals.copy_done += read.copy.tick <= read.completed ? 1 : 0;
    totals.early_copies += read.copy.early ? 1 : 0;
    totals.lock += read.locked - read.woke;
    totals.land += read.landed - read.locked;
    totals.backlog_subs += read.submitted > read.completed ? read.submitted - read.completed : 0;
    totals.unqueued_subs +=
        read.recording > read.submitted + 1 ? read.recording - read.submitted - 1 : 0;
    if (timeline.has_submit) {
        ++totals.submits;
        totals.submit += timeline.submit;
    }
    if (timeline.placed) {
        ++totals.placed;
        totals.to_write += timeline.to_write;
        totals.to_copy += timeline.to_copy;
        totals.copy += timeline.copy;
        totals.to_wake += timeline.to_wake;
        totals.idle_to_write += timeline.idle_to_write;
        totals.idle_to_copy += timeline.idle_to_copy;
        totals.backlog_busy += timeline.backlog_busy;
        totals.uncertainty += read.uncertainty;
        totals.write_ended += timeline.write_lead != 0 ? 1 : 0;
        totals.write_lead += timeline.write_lead;
        totals.write_gap += timeline.write_gap;
    }
    if (!read.has_write) {
        ++totals.no_write;
        return;
    }
    const Write& write = read.write;
    if (write.tick <= read.completed) {
        ++totals.write_done;
    } else if (write.tick <= read.submitted) {
        ++totals.write_on_gpu;
    } else if (write.tick < read.recording) {
        ++totals.write_unqueued;
    } else {
        ++totals.write_recording;
    }
    if (write.recorded <= read.request) {
        ++totals.written_before;
        totals.since_write += read.request - write.recorded;
    }
    const Counts& after = read.copy.counts;
    totals.draws += after.draws - write.counts.draws;
    totals.dispatches += after.dispatches - write.counts.dispatches;
    totals.render_passes += after.render_passes - write.counts.render_passes;
    totals.submissions += read.copy.tick - write.tick;
}

inline std::string Format(u64 area, const Totals& t) {
    const auto ms = [](u64 sum, u64 count) {
        return count == 0 ? 0.0 : static_cast<double>(sum) / 1e6 / static_cast<double>(count);
    };
    const auto per = [](u64 sum, u64 count) {
        return count == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(count);
    };
    std::string line = area == 0 ? std::string{"all"} : fmt::format("{:#x}", area);
    const u64 copied = t.reads - t.no_copy;
    line += fmt::format(": {} reads ({} in flight, {} waited for an early download, {} without a "
                        "copy, {} with it done at the request, {} placed on the GPU timeline "
                        "+-{:.3f}), wait {:.2f}",
                        t.reads, t.in_flight, t.early_copies, t.no_copy, t.copy_done, t.placed,
                        ms(t.uncertainty, t.placed) / 2, ms(t.wait, t.reads));
    if (t.placed != 0) {
        line += fmt::format(" | from the request: to the write's end {:.2f} (GPU idle {:.2f}), to "
                            "the copy's start {:.2f} (idle {:.2f}), copy {:.2f}, to the wake-up "
                            "{:.2f}",
                            ms(t.to_write, t.placed), ms(t.idle_to_write, t.placed),
                            ms(t.to_copy, t.placed), ms(t.idle_to_copy, t.placed),
                            ms(t.copy, t.placed), ms(t.to_wake, t.placed));
    }
    if (copied != 0) {
        line += fmt::format("{}lock {:.2f}, to guest memory {:.2f}", t.placed != 0 ? ", " : " | ",
                            ms(t.lock, copied), ms(t.land, copied));
    }
    if (t.placed != 0) {
        line +=
            fmt::format("; the write's end to the copy's start {:.2f}; the write had ended "
                        "at the request {}x, {:.2f} before it",
                        ms(t.write_gap, t.placed), t.write_ended, ms(t.write_lead, t.write_ended));
    }
    if (copied != 0) {
        line += fmt::format(" | at the request: {:.1f} submissions in flight",
                            per(t.backlog_subs, copied));
        if (t.placed != 0) {
            line += fmt::format(" (GPU busy {:.2f} more)", ms(t.backlog_busy, t.placed));
        }
        line += fmt::format(", {:.1f} flushed but not queued; the copy queued {:.3f} after its "
                            "flush",
                            per(t.unqueued_subs, copied), ms(t.submit, t.submits));
    }
    const u64 written = copied - t.no_write;
    if (written != 0) {
        line += fmt::format(" | the write at the request: done {}, on the GPU {}, flushed but not "
                            "queued {}, still recording {}; recorded {:.2f} before it ({} after); "
                            "then {:.1f} draws, {:.1f} dispatches, {:.1f} render passes, {:.1f} "
                            "submissions until the copy",
                            t.write_done, t.write_on_gpu, t.write_unqueued, t.write_recording,
                            ms(t.since_write, t.written_before), written - t.written_before,
                            per(t.draws, written), per(t.dispatches, written),
                            per(t.render_passes, written), per(t.submissions, written));
    }
    if (t.no_write != 0) {
        line += fmt::format(" ({} without a write seen)", t.no_write);
    }
    if (t.sampled != 0) {
        line +=
            fmt::format(" | sampled frames, {} reads: draws and dispatches {:.2f} of the {:.2f} "
                        "busy to the write's end, {:.2f} of {:.2f} to the copy's start",
                        t.sampled, ms(t.timed_to_write, t.sampled), ms(t.busy_to_write, t.sampled),
                        ms(t.timed_to_copy, t.sampled), ms(t.busy_to_copy, t.sampled));
    }
    return line;
}

/// `mutex` held.
inline void MaybeReport(u64 now) {
    const u64 presents = GpuWaitStats::g_presents.load(std::memory_order_relaxed);
    if (g_state.window_start == 0) {
        g_state.window_start = now;
        g_state.presents = presents;
        return;
    }
    if (now - g_state.window_start < GpuWaitStats::WindowNs) {
        return;
    }
    std::ranges::sort(g_state.areas, std::greater<>{},
                      [](const auto& entry) { return entry.second.wait; });
    LOG_INFO(HW_GPU, "Read trace (EDEN_NX2_READ_TRACE) in {:.1f} s, {} frames: {}",
             static_cast<double>(now - g_state.window_start) / 1e9, presents - g_state.presents,
             Format(0, g_state.all));
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 gpu_side_waits = take(g_early.gpu_side_waits);
    LOG_INFO(HW_GPU,
             "Read trace early downloads: {} recorded ({} KiB) | reads they served: {} found "
             "the bytes in guest memory, {} found them done on the host GPU (landed at the read), "
             "{} waited for them (in the lines above) | writes from the GPU's side that waited "
             "for one: {} ({:.2f} ms)",
             take(g_early.recorded), take(g_early.bytes) / 1024, take(g_early.served_landed),
             take(g_early.served_at_read), take(g_early.served_in_flight), gpu_side_waits,
             static_cast<double>(take(g_early.gpu_side_wait_ns)) / 1e6);
    for (size_t i = 0; i < g_state.areas.size() && i < ShownAreas; ++i) {
        LOG_INFO(HW_GPU, "Read trace area {}",
                 Format(g_state.areas[i].first, g_state.areas[i].second));
    }
    g_state.window_start = now;
    g_state.presents = presents;
    g_state.all = {};
    g_state.areas.clear();
    g_state.other_areas = {};
}

/// The reading thread, once its read is done (from Vulkan::Scheduler).
inline void Record(const Read& read, const Timeline& timeline) {
    std::scoped_lock lock{g_state.mutex};
    Add(g_state.all, read, timeline);
    Add(AreaTotals(read.area), read, timeline);
    if (timeline.placed) {
        g_state.windows[g_state.next_window++ % g_state.windows.size()] = Window{
            .area = read.area,
            .ticks = timeline.window,
            .busy_to_write = timeline.to_write - timeline.idle_to_write,
            .busy_to_copy = timeline.to_copy - timeline.idle_to_copy,
            .pending = true,
        };
    }
    MaybeReport(Now());
}

/// Once a frame (BufferCache::TickFrame): reports even while no read waits.
inline void Tick() {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{g_state.mutex};
    MaybeReport(Now());
}

/// The Vulkan worker, with the draws and dispatches of a frame EDEN_DKB_FRAME_TRACE timed
/// ([begin, end] device ticks, in order): splits the GPU time of the reads placed in it.
inline void NoteTimedWork(std::span<const std::pair<u64, u64>> work, double ns_per_tick) {
    if (work.empty()) {
        return;
    }
    const u64 first = work.front().first;
    const u64 last = work.back().second;
    const auto timed = [&](u64 lower, u64 upper) {
        u64 ticks = 0;
        for (const auto& [begin, end] : work) {
            const u64 from = (std::max)(begin, lower);
            const u64 to = (std::min)(end, upper);
            ticks += to > from ? to - from : 0;
        }
        return static_cast<u64>(static_cast<double>(ticks) * ns_per_tick);
    };
    std::scoped_lock lock{g_state.mutex};
    for (Window& window : g_state.windows) {
        if (!window.pending || window.ticks[0] < first || window.ticks[2] > last) {
            continue;
        }
        window.pending = false;
        const u64 timed_to_write = timed(window.ticks[0], window.ticks[1]);
        const u64 timed_to_copy = timed(window.ticks[1], window.ticks[2]);
        for (Totals* totals : {&g_state.all, &AreaTotals(window.area)}) {
            ++totals->sampled;
            totals->timed_to_write += timed_to_write;
            totals->busy_to_write += window.busy_to_write;
            totals->timed_to_copy += timed_to_copy;
            totals->busy_to_copy += window.busy_to_copy;
        }
    }
}

} // namespace VideoCommon::ReadTrace
