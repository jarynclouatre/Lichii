// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/cpu_features.h"
#include "common/logging.h"
#include "core/hardware_properties.h"
#include "video_core/rasterizer_download_area.h"

// TEST: how long guest threads wait for the GPU. Always on: one line every 10 s ("GPU waits").
//
// A guest read of GPU-written memory that GetFlushArea can't serve (reactive flushing) queues a
// flush on the GPU thread and waits for it. Each wait is timed from the request to the wake-up and
// split into: queue (until the GPU thread starts the flush), flush (applying the collected CPU
// writes, then the downloads; "host GPU" is the time in Vulkan's Scheduler::Finish, waiting for
// the host GPU), wake (until the waiting thread runs again) and the copies (the waiting thread's
// own wait for the downloads the flush submitted, and writing them to guest memory: FinishRead).
// Also counted: guest reads and writes that take the rasterizer path, by core, with the time of
// their cache lookups (GetFlushArea, OnCPUWrite: both take the cache locks the GPU thread holds
// while it draws); the other sync operations guest threads wait for; the GPU thread's (and the
// split's back thread's) idle time; every Scheduler::Finish and every wait for pending query
// results (FlushQueryResults); the time between presents; and, for the pages that waited longest,
// the GPU writes that marked them.
namespace VideoCommon::GpuWaitStats {

/// One slot per guest core, the last one for every other thread.
inline constexpr size_t CoreSlots = Core::Hardware::NUM_CPU_CORES + 1;
/// Pages whose GPU writers are recorded (the longest waits of the last window).
inline constexpr size_t WatchSlots = 4;
/// Distinct flush areas totaled per window.
inline constexpr size_t MaxAreas = 64;
inline constexpr u64 WindowNs = 10'000'000'000ULL;
inline constexpr u64 PageSize = 0x1000;

/// Steady nanoseconds: the time stamp counter where the wall clock reads it directly, else
/// steady_clock (as Nx2Profile::Now). Cheap enough for the per-read counters.
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

inline void AtomicMax(std::atomic<u64>& value, u64 candidate) {
    u64 current = value.load(std::memory_order_relaxed);
    while (candidate > current &&
           !value.compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {
    }
}

/// Increments a counter only one thread writes (no locked instruction).
inline void BumpOwned(std::atomic<u64>& value, u64 amount = 1) {
    value.store(value.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

/// What marked buffer memory GPU-written (BufferCache::MarkWrittenBuffer and DMACopy).
enum class WriteSite : u8 {
    Other,
    GraphicsStorage,
    GraphicsTexel,
    ComputeStorage,
    ComputeTexel,
    MultiRangeStorage,
    TransformFeedback,
    ImageToBuffer, ///< ObtainBuffer with MarkAsWritten: DMA copies of images into buffers
    DmaCopy,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(WriteSite::Count)> write_site_names{
    "other",
    "graphics storage",
    "graphics texel buffer",
    "compute storage",
    "compute texel buffer",
    "multi-range storage",
    "transform feedback",
    "image to buffer",
    "DMA copy",
};

inline constexpr std::array<const char*, 3> source_names{"other", "texture", "buffer"};

/// Sync operations other than reactive flushes that a guest thread waits for.
enum class Site : u8 {
    Invalidate, ///< A CPU write over GPU-written buffer memory (InvalidateRegion, preserved)
    Idle,       ///< GPU::WaitForIdle
    Count,
};

/// Filled by the GPU thread while the guest thread waits for it.
struct FlushTiming {
    u64 start{};   ///< The GPU thread started the flush
    u64 applied{}; ///< It applied the CPU writes collected so far
    u64 end{};
    u64 host_ns{}; ///< Of the flush: time in Scheduler::Finish
    u64 finishes{};
    u64 bytes{}; ///< Downloaded by the flush
    /// The reading thread, after the flush: waiting for its downloads and writing them to guest
    /// memory (RasterizerInterface::FinishRead)
    u64 finish_ns{};
    bool in_flight{}; ///< No flush: only waited for an earlier read's downloads
};

struct HostWaits {
    u64 ns{};
    u64 count{};
};

/// This thread's Scheduler::Finish totals, so a flush can tell its own host waits.
inline thread_local HostWaits t_host_waits{};
inline std::atomic<u64> g_host_wait_ns{};
inline std::atomic<u64> g_host_waits{};

/// Scheduler::Finish took `ns` (submit, then wait for the host GPU).
inline void AddHostWait(u64 ns) {
    t_host_waits.ns += ns;
    ++t_host_waits.count;
    g_host_wait_ns.fetch_add(ns, std::memory_order_relaxed);
    g_host_waits.fetch_add(1, std::memory_order_relaxed);
}

/// Bytes this thread's buffer and texture cache downloads copied back (read as a difference by a
/// reactive flush, like t_host_waits).
inline thread_local u64 t_download_bytes{};

inline void AddDownloadBytes(u64 bytes) {
    t_download_bytes += bytes;
}

/// RasterizerVulkan::FlushQueryResults waited `ns` for pending query results (macro DMA reads of
/// report regions: the GPU thread waits for the host GPU to reach a new fence).
inline std::atomic<u64> g_query_wait_ns{};
inline std::atomic<u64> g_query_waits{};

inline void AddQueryWait(u64 ns) {
    g_query_wait_ns.fetch_add(ns, std::memory_order_relaxed);
    g_query_waits.fetch_add(1, std::memory_order_relaxed);
}

struct alignas(64) CoreCounter {
    std::atomic<u64> value{};
};
/// Guest reads that took the rasterizer path (Memory HandleRasterizerDownload), by core.
inline std::array<CoreCounter, CoreSlots> g_rasterizer_reads{};

inline void CountRasterizerRead(size_t core) {
    if (core < CoreSlots - 1) {
        BumpOwned(g_rasterizer_reads[core].value);
    } else {
        g_rasterizer_reads[CoreSlots - 1].value.fetch_add(1, std::memory_order_relaxed);
    }
}

/// Guest writes that took the rasterizer path (Memory HandleRasterizerWrite), and their
/// GPU::OnCPUWrite calls (the buffer and texture cache locks) with time, by core. As in Memory,
/// the last core's slot also takes the other threads, under Memory's guard: one writer at a time.
inline std::array<CoreCounter, Core::Hardware::NUM_CPU_CORES> g_rasterizer_writes{};
inline std::array<CoreCounter, Core::Hardware::NUM_CPU_CORES> g_write_checks{};
inline std::array<CoreCounter, Core::Hardware::NUM_CPU_CORES> g_write_check_ns{};

inline void CountRasterizerWrite(size_t core) {
    if (core < g_rasterizer_writes.size()) {
        BumpOwned(g_rasterizer_writes[core].value);
    }
}

inline void AddWriteCheck(size_t core, u64 ns) {
    if (core < g_write_checks.size()) {
        BumpOwned(g_write_checks[core].value);
        BumpOwned(g_write_check_ns[core].value, ns);
    }
}

/// GPU::OnCPURead's GetFlushArea calls (served or not) and their time (two cache locks), by core
/// (GetCurrentHostThreadID; the last slot takes every other thread).
inline std::array<CoreCounter, CoreSlots> g_area_checks{};
inline std::array<CoreCounter, CoreSlots> g_area_check_ns{};

inline void AddAreaCheck(size_t core, u64 ns) {
    if (core < CoreSlots - 1) {
        BumpOwned(g_area_checks[core].value);
        BumpOwned(g_area_check_ns[core].value, ns);
    } else {
        g_area_checks[CoreSlots - 1].value.fetch_add(1, std::memory_order_relaxed);
        g_area_check_ns[CoreSlots - 1].value.fetch_add(ns, std::memory_order_relaxed);
    }
}

/// Idle time of the GPU thread (waiting for commands; polling suspended channels) and of the
/// split's back thread (waiting for the front).
inline std::atomic<u64> g_front_idle_ns{};
inline std::atomic<u64> g_front_poll_ns{};
inline std::atomic<u64> g_back_idle_ns{};

class IdleScope {
public:
    explicit IdleScope(std::atomic<u64>& sink_) : sink{sink_}, start{Now()} {}
    ~IdleScope() {
        sink.fetch_add(Now() - start, std::memory_order_relaxed);
    }
    IdleScope(const IdleScope&) = delete;
    IdleScope& operator=(const IdleScope&) = delete;

private:
    std::atomic<u64>& sink;
    u64 start;
};

struct SyncTotals {
    std::atomic<u64> count{};
    std::atomic<u64> ns{};
};
inline std::array<SyncTotals, static_cast<size_t>(Site::Count)> g_syncs{};

inline void RecordSync(Site site, u64 ns) {
    auto& totals = g_syncs[static_cast<size_t>(site)];
    totals.count.fetch_add(1, std::memory_order_relaxed);
    totals.ns.fetch_add(ns, std::memory_order_relaxed);
}

/// Time between presents (nx2:vi present).
inline std::atomic<u64> g_last_present{};
inline std::atomic<u64> g_presents{};
inline std::atomic<u64> g_present_ns{};
inline std::atomic<u64> g_present_max_ns{};

inline void NotePresent() {
    const u64 now = Now();
    const u64 last = g_last_present.exchange(now, std::memory_order_relaxed);
    if (last == 0 || now <= last) {
        return;
    }
    g_presents.fetch_add(1, std::memory_order_relaxed);
    g_present_ns.fetch_add(now - last, std::memory_order_relaxed);
    AtomicMax(g_present_max_ns, now - last);
}

/// GPU writes over a watched page: written by the buffer cache under its lock, read by the report.
struct alignas(64) Watch {
    std::atomic<u64> page{}; ///< 0: unused
    std::atomic<u64> writes{};
    std::atomic<u64> last_start{};
    std::atomic<u64> last_size{};
    std::atomic<u64> largest{};
    std::atomic<u32> last_site{};
};
inline std::array<Watch, WatchSlots> g_watches{};
inline std::atomic<u32> g_watching{};

/// The buffer cache marked [addr, addr + size) GPU-written.
inline void NoteGpuWrite(u64 addr, u64 size, WriteSite site) {
    if (g_watching.load(std::memory_order_relaxed) == 0) {
        return;
    }
    for (Watch& watch : g_watches) {
        const u64 page = watch.page.load(std::memory_order_relaxed);
        if (page == 0 || page >= addr + size || addr >= page + PageSize) {
            continue;
        }
        BumpOwned(watch.writes);
        watch.last_start.store(addr, std::memory_order_relaxed);
        watch.last_size.store(size, std::memory_order_relaxed);
        watch.last_site.store(static_cast<u32>(site), std::memory_order_relaxed);
        if (size > watch.largest.load(std::memory_order_relaxed)) {
            watch.largest.store(size, std::memory_order_relaxed);
        }
    }
}

struct AreaTotals {
    u64 start{};
    u64 size{};
    u64 count{};
    u64 wait_ns{};
    u64 max_ns{};
    VideoCore::DownloadAreaSource source{};
};

struct ReactiveTotals {
    u64 count{};
    u64 wait_ns{};
    u64 max_ns{};
    u64 queue_ns{};
    u64 apply_ns{};
    u64 flush_ns{};
    u64 host_ns{};
    u64 finishes{};
    u64 without_finish{};
    u64 bytes{};
    u64 max_bytes{};
    u64 wake_ns{};
    u64 finish_ns{};
    u64 in_flight{};
    u64 other_areas{}; ///< Flushes of areas past MaxAreas
    std::array<u64, 3> count_by_source{};
    std::array<u64, 3> ns_by_source{};
    std::array<u64, CoreSlots> ns_by_core{};
    std::vector<AreaTotals> areas;
};
inline std::mutex g_reactive_mutex;
inline ReactiveTotals g_reactive;

/// A guest thread (`core`, GetCurrentHostThreadID) waited from `request` to `woke` for the flush
/// of `area`, which the GPU thread ran as `timing` says.
inline void RecordReactive(const VideoCore::RasterizerDownloadArea& area, size_t core, u64 request,
                           const FlushTiming& timing, u64 woke) {
    const u64 wait = woke - request;
    const size_t source = (std::min<size_t>)(static_cast<size_t>(area.source), 2);
    std::scoped_lock lock{g_reactive_mutex};
    ReactiveTotals& totals = g_reactive;
    ++totals.count;
    totals.wait_ns += wait;
    totals.max_ns = (std::max)(totals.max_ns, wait);
    if (timing.start >= request && timing.end >= timing.start &&
        woke >= timing.end + timing.finish_ns) {
        totals.queue_ns += timing.start - request;
        totals.apply_ns += timing.applied - timing.start;
        totals.flush_ns += timing.end - timing.start;
        totals.wake_ns += woke - timing.end - timing.finish_ns;
    }
    totals.finish_ns += timing.finish_ns;
    totals.in_flight += timing.in_flight ? 1 : 0;
    totals.host_ns += timing.host_ns;
    totals.finishes += timing.finishes;
    totals.without_finish += timing.finishes == 0 ? 1 : 0;
    totals.bytes += timing.bytes;
    totals.max_bytes = (std::max)(totals.max_bytes, timing.bytes);
    ++totals.count_by_source[source];
    totals.ns_by_source[source] += wait;
    totals.ns_by_core[(std::min)(core, CoreSlots - 1)] += wait;
    const auto it =
        std::find_if(totals.areas.begin(), totals.areas.end(), [&area](const AreaTotals& entry) {
            return entry.start == area.start_address;
        });
    AreaTotals* entry = it != totals.areas.end() ? &*it : nullptr;
    if (entry == nullptr) {
        if (totals.areas.size() >= MaxAreas) {
            ++totals.other_areas;
            return;
        }
        entry = &totals.areas.emplace_back();
        entry->start = area.start_address;
        entry->source = area.source;
    }
    entry->size = (std::max<u64>)(entry->size, area.end_address - area.start_address);
    ++entry->count;
    entry->wait_ns += wait;
    entry->max_ns = (std::max)(entry->max_ns, wait);
}

/// Report state: the cumulative counters at the start of the window (GPU thread only).
struct ReportState {
    u64 window_start{};
    std::array<u64, CoreSlots> reads{};
    std::array<u64, Core::Hardware::NUM_CPU_CORES> writes{};
    std::array<u64, Core::Hardware::NUM_CPU_CORES> write_checks{};
    std::array<u64, Core::Hardware::NUM_CPU_CORES> write_check_ns{};
    u64 area_checks{};
    u64 area_check_ns{};
    u64 front_idle{};
    u64 front_poll{};
    u64 back_idle{};
    u64 host_waits{};
    u64 host_wait_ns{};
    u64 query_waits{};
    u64 query_wait_ns{};
    std::array<u64, static_cast<size_t>(Site::Count)> sync_count{};
    std::array<u64, static_cast<size_t>(Site::Count)> sync_ns{};
    u64 presents{};
    u64 present_ns{};
};
inline ReportState g_report;

/// Takes the cumulative counters' current values; returns the old ones.
inline ReportState Snapshot(u64 now) {
    ReportState state;
    state.window_start = now;
    for (size_t i = 0; i < CoreSlots; ++i) {
        state.reads[i] = g_rasterizer_reads[i].value.load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < state.writes.size(); ++i) {
        state.writes[i] = g_rasterizer_writes[i].value.load(std::memory_order_relaxed);
        state.write_checks[i] = g_write_checks[i].value.load(std::memory_order_relaxed);
        state.write_check_ns[i] = g_write_check_ns[i].value.load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < CoreSlots; ++i) {
        state.area_checks += g_area_checks[i].value.load(std::memory_order_relaxed);
        state.area_check_ns += g_area_check_ns[i].value.load(std::memory_order_relaxed);
    }
    state.front_idle = g_front_idle_ns.load(std::memory_order_relaxed);
    state.front_poll = g_front_poll_ns.load(std::memory_order_relaxed);
    state.back_idle = g_back_idle_ns.load(std::memory_order_relaxed);
    state.host_waits = g_host_waits.load(std::memory_order_relaxed);
    state.host_wait_ns = g_host_wait_ns.load(std::memory_order_relaxed);
    state.query_waits = g_query_waits.load(std::memory_order_relaxed);
    state.query_wait_ns = g_query_wait_ns.load(std::memory_order_relaxed);
    for (size_t i = 0; i < state.sync_count.size(); ++i) {
        state.sync_count[i] = g_syncs[i].count.load(std::memory_order_relaxed);
        state.sync_ns[i] = g_syncs[i].ns.load(std::memory_order_relaxed);
    }
    state.presents = g_presents.load(std::memory_order_relaxed);
    state.present_ns = g_present_ns.load(std::memory_order_relaxed);
    return std::exchange(g_report, state);
}

inline double Ms(u64 ns) {
    return static_cast<double>(ns) / 1e6;
}

/// Logs the window's totals every 10 s. Called by the GPU thread between commands, so its own
/// idle time is complete; `split_active` says whether the back thread's counts apply.
inline void MaybeReport(bool split_active) {
    const u64 now = Now();
    if (g_report.window_start == 0) {
        (void)Snapshot(now);
        return;
    }
    if (now - g_report.window_start < WindowNs) {
        return;
    }
    const ReportState old = Snapshot(now);
    const ReportState& cur = g_report;
    const u64 window = now - old.window_start;
    ReactiveTotals reactive;
    {
        std::scoped_lock lock{g_reactive_mutex};
        reactive = std::move(g_reactive);
        g_reactive = {};
    }
    const u64 presents = cur.presents - old.presents;
    const u64 present_ns = cur.present_ns - old.present_ns;
    const u64 present_max = g_present_max_ns.exchange(0, std::memory_order_relaxed);

    std::string line = fmt::format("GPU waits in {:.1f} s:", static_cast<double>(window) / 1e9);
    if (presents != 0) {
        line += fmt::format(" {} frames ({:.1f}/s, {:.2f} ms each, max {:.2f} ms) |", presents,
                            static_cast<double>(presents) * 1e9 / static_cast<double>(window),
                            Ms(present_ns) / static_cast<double>(presents), Ms(present_max));
    } else {
        line += " no presents |";
    }
    line +=
        fmt::format(" reactive flushes {} waited {:.1f} ms ({:.1f}% of the time", reactive.count,
                    Ms(reactive.wait_ns),
                    static_cast<double>(reactive.wait_ns) * 100.0 / static_cast<double>(window));
    if (presents != 0) {
        line += fmt::format(", {:.2f} ms and {:.2f} flushes a frame",
                            Ms(reactive.wait_ns) / static_cast<double>(presents),
                            static_cast<double>(reactive.count) / static_cast<double>(presents));
    }
    line += fmt::format("), max {:.2f} ms", Ms(reactive.max_ns));
    // Ends a list of ", "-separated items: drops the separator after the last one.
    const auto end_list = [&line] {
        if (line.size() >= 1 && line.back() == ',') {
            line.pop_back();
        }
    };
    if (reactive.count != 0) {
        line += "; by cache:";
        for (size_t i = 0; i < source_names.size(); ++i) {
            if (reactive.count_by_source[i] != 0) {
                line += fmt::format(" {} {} {:.1f} ms,", source_names[i],
                                    reactive.count_by_source[i], Ms(reactive.ns_by_source[i]));
            }
        }
        end_list();
        line += fmt::format(" | queue {:.1f} ms, flush {:.1f} ms (CPU writes {:.1f}, host GPU "
                            "{:.1f} in {} finishes, {} flushes without one; downloaded {} KiB, "
                            "at most {} KiB), wake {:.1f} ms, then copies on the reading thread "
                            "{:.1f} ms ({} reads only waited for an earlier read's) | by core:",
                            Ms(reactive.queue_ns), Ms(reactive.flush_ns), Ms(reactive.apply_ns),
                            Ms(reactive.host_ns), reactive.finishes, reactive.without_finish,
                            reactive.bytes / 1024, reactive.max_bytes / 1024, Ms(reactive.wake_ns),
                            Ms(reactive.finish_ns), reactive.in_flight);
        for (size_t i = 0; i < CoreSlots; ++i) {
            if (reactive.ns_by_core[i] == 0) {
                continue;
            }
            line += i + 1 < CoreSlots
                        ? fmt::format(" {} {:.1f} ms,", i, Ms(reactive.ns_by_core[i]))
                        : fmt::format(" other {:.1f} ms,", Ms(reactive.ns_by_core[i]));
        }
        end_list();
    }
    // The longest waits, with the GPU writes seen over them (watched since the last report).
    std::sort(reactive.areas.begin(), reactive.areas.end(),
              [](const AreaTotals& a, const AreaTotals& b) { return a.wait_ns > b.wait_ns; });
    line += " | areas:";
    const size_t shown = (std::min)(reactive.areas.size(), WatchSlots);
    if (shown == 0) {
        line += " none";
    }
    for (size_t i = 0; i < shown; ++i) {
        const AreaTotals& area = reactive.areas[i];
        line += fmt::format(" {:#x}+{:#x} {} {}x {:.1f} ms (max {:.2f})", area.start, area.size,
                            source_names[(std::min<size_t>)(static_cast<size_t>(area.source), 2)],
                            area.count, Ms(area.wait_ns), Ms(area.max_ns));
        for (const Watch& watch : g_watches) {
            if (watch.page.load(std::memory_order_relaxed) != area.start) {
                continue;
            }
            const u64 writes = watch.writes.load(std::memory_order_relaxed);
            if (writes == 0) {
                line += " no GPU writes seen";
            } else {
                const u32 site = (std::min<u32>)(watch.last_site.load(std::memory_order_relaxed),
                                                 static_cast<u32>(WriteSite::Count) - 1);
                line += fmt::format(" GPU-written {}x, last by {} {:#x}+{:#x} (largest {:#x})",
                                    writes, write_site_names[site],
                                    watch.last_start.load(std::memory_order_relaxed),
                                    watch.last_size.load(std::memory_order_relaxed),
                                    watch.largest.load(std::memory_order_relaxed));
            }
        }
        line += ',';
    }
    if (reactive.areas.size() > shown || reactive.other_areas != 0) {
        line +=
            fmt::format(" {} more areas,", reactive.areas.size() - shown + reactive.other_areas);
    }
    end_list();
    // Watch the longest waits' buffer pages during the next window.
    g_watching.store(0, std::memory_order_relaxed);
    u32 watching = 0;
    for (size_t i = 0; i < WatchSlots; ++i) {
        Watch& watch = g_watches[i];
        u64 page = 0;
        if (i < shown && reactive.areas[i].source == VideoCore::DownloadAreaSource::BufferCache) {
            page = reactive.areas[i].start;
        }
        watch.page.store(0, std::memory_order_relaxed);
        watch.writes.store(0, std::memory_order_relaxed);
        watch.last_start.store(0, std::memory_order_relaxed);
        watch.last_size.store(0, std::memory_order_relaxed);
        watch.largest.store(0, std::memory_order_relaxed);
        watch.last_site.store(0, std::memory_order_relaxed);
        watch.page.store(page, std::memory_order_relaxed);
        watching += page != 0 ? 1 : 0;
    }
    g_watching.store(watching, std::memory_order_relaxed);

    line += " | other waits:";
    constexpr std::array<const char*, static_cast<size_t>(Site::Count)> site_names{"invalidate",
                                                                                   "idle"};
    for (size_t i = 0; i < site_names.size(); ++i) {
        line +=
            fmt::format(" {} {}x {:.1f} ms,", site_names[i], cur.sync_count[i] - old.sync_count[i],
                        Ms(cur.sync_ns[i] - old.sync_ns[i]));
    }
    end_list();
    line += " | rasterizer-path reads by core:";
    u64 all_reads = 0;
    for (size_t i = 0; i < CoreSlots; ++i) {
        const u64 reads = cur.reads[i] - old.reads[i];
        all_reads += reads;
        if (reads != 0) {
            line += i + 1 < CoreSlots ? fmt::format(" {} {},", i, reads)
                                      : fmt::format(" other {},", reads);
        }
    }
    if (all_reads == 0) {
        line += " none";
    }
    end_list();
    line += fmt::format("; flush area checks {} in {:.1f} ms", cur.area_checks - old.area_checks,
                        Ms(cur.area_check_ns - old.area_check_ns));
    line += " | rasterizer-path writes by core (OnCPUWrite calls, time):";
    u64 all_writes = 0;
    for (size_t i = 0; i < cur.writes.size(); ++i) {
        const u64 writes = cur.writes[i] - old.writes[i];
        all_writes += writes;
        if (writes != 0) {
            line += fmt::format(" {} {} ({}, {:.1f} ms),", i, writes,
                                cur.write_checks[i] - old.write_checks[i],
                                Ms(cur.write_check_ns[i] - old.write_check_ns[i]));
        }
    }
    if (all_writes == 0) {
        line += " none";
    }
    end_list();
    const auto percent = [window](u64 ns) {
        return static_cast<double>(ns) * 100.0 / static_cast<double>(window);
    };
    const u64 front_idle = cur.front_idle - old.front_idle;
    const u64 front_poll = cur.front_poll - old.front_poll;
    line += fmt::format(" | GPU thread busy {:.1f}% (idle {:.1f}%, polling suspended channels "
                        "{:.1f}%)",
                        100.0 - percent(front_idle + front_poll), percent(front_idle),
                        percent(front_poll));
    if (split_active) {
        line += fmt::format(", back thread busy {:.1f}%",
                            100.0 - percent(cur.back_idle - old.back_idle));
    }
    line +=
        fmt::format(" | Scheduler::Finish {}x {:.1f} ms; query result waits {}x {:.1f} ms",
                    cur.host_waits - old.host_waits, Ms(cur.host_wait_ns - old.host_wait_ns),
                    cur.query_waits - old.query_waits, Ms(cur.query_wait_ns - old.query_wait_ns));
    LOG_INFO(HW_GPU, "{}", line);
}

} // namespace VideoCommon::GpuWaitStats
