// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#ifdef _MSC_VER
#include <intrin.h>
#elif defined(ARCHITECTURE_x86_64)
#include <x86intrin.h>
#endif
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_set>

#include "common/common_types.h"
#include "video_core/gpu_split.h"
#include "common/logging.h"
#include "common/cpu_features.h"
#include "common/thread_cpu_clock.h"

// EDEN_NX2_GPU_PROFILE: time where the GPU thread spends its time. Each zone counts its own
// (exclusive) time, nested zones excluded; every 10 s the thread logs each zone's share.
namespace VideoCommon::Nx2Profile {

enum class Zone : u32 {
    Other, ///< GPU thread time outside every zone below (register writes, command parsing)
    Idle,
    PendingSleep,
    ResumePending,
    Push,
    Tick,
    Run,
    FlushInvalidate,
    Macro,
    Draw,
    Clear,
    Compute,
    FlushWork,
    PipelineLookup,
    Configure,
    SyncDescriptors,
    ReadHandles,
    FillImageViews,
    TextureBuffers,
    UpdateBuffers,
    DescriptorAcquire,
    BindStageBuffers,
    PushDescriptors,
    RenderTargets,
    ConfigureDraw,
    DynamicState,
    DrawRecord,
    TextureUpload,
    Fence,        ///< Fences made (SignalFence)
    FenceSkipped, ///< Empty fences not made
    Method,       ///< Engine method calls (what a split GPU thread's back half would run)
    SplitDrain,   ///< Split GPU thread: the front waits for the back (gpu_split.h)
    SplitWait,    ///< Split GPU thread: the back waits for work
    WriterWait,
    ListSubmitWait,
    ListGpuWait,
    AcquirePoll, ///< TEST: a suspended acquire's poll (CanResumeAcquire in ResumePending)
    Resnapshot,  ///< TEST: command lists copied again at a resume (DmaPusher::RefreshSnapshot)
    // TEST: fine zones, EDEN_NX2_GPU_PROFILE=2 only (FineScope): they split the zones above.
    CbBind,          ///< Of method: constant buffer binds (Maxwell3D::ProcessCBBind)
    CbUpload,        ///< Of method: constant buffer words written to memory
    QueryReport,     ///< Of method: report semaphores (Maxwell3D::ProcessQueryGet)
    CounterReset,    ///< Of method: counter resets (Maxwell3D::ProcessCounterReset)
    RenderEnable,    ///< Of method: render enable conditions (Maxwell3D::ProcessQueryCondition)
    Sink,            ///< State writes held for the next executable method (ConsumeSinkImpl)
    MacroEnd,        ///< A front macro's end on the back (Maxwell3D::FinishFrontMacro)
    DrawSetup,       ///< Of draw: after the pipeline lookup, before Configure (cache locks)
    DrawQueries,     ///< Of draw: query and transform feedback state before recording
    TickWork,        ///< Of draw and compute: the sync operations after them (GPU::TickWork)
    UniformBuffers,  ///< Of bind stage buffers: the uniform buffers
    GeometryBuffers, ///< Of update buffers: index and vertex buffer binds
    StorageBindings, ///< Of read handles: storage buffer addresses from constant buffers
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Zone::Count)> zone_names{
    "other",         "idle",          "pending sleep",  "resume pending", "push",
    "tick",          "run",           "flush/inval",    "macro",          "draw",
    "clear",         "compute",       "flush work",     "pipeline lookup", "configure",
    "sync descriptors", "read handles", "fill image views", "texture buffers", "update buffers",
    "descriptor acquire", "bind stage buffers", "push descriptors", "render targets",
    "configure draw", "dynamic state", "draw record", "texture upload", "fence",
    "fence skipped", "method", "split drain", "split wait",
    "writer wait", "list submit wait", "list GPU wait", "acquire poll", "resnapshot",
    "cb bind", "cb upload", "query report", "counter reset", "render enable", "sink",
    "macro end", "draw setup", "draw queries", "tick work", "uniform buffers",
    "geometry buffers", "storage bindings",
};
static_assert(zone_names.back() != nullptr, "a zone without a name");

// A plain flag: Enabled() runs several times per method write (no guard to check).
inline bool g_enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr;

/// TEST: EDEN_NX2_GPU_PROFILE=2 adds the fine zones (FineScope) and counters (FineCounter), at
/// the cost of more time stamp reads: =1 keeps the zones runs compare with.
[[nodiscard]] inline bool FineSwitchSet() {
    const char* value = std::getenv("EDEN_NX2_GPU_PROFILE");
    return value != nullptr && value[0] == '2';
}
inline bool g_fine = FineSwitchSet();

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr;
    g_fine = FineSwitchSet();
}

inline bool Enabled() {
    return g_enabled;
}

inline bool Fine() {
    return g_fine;
}

// Nanoseconds; from the host's own counter where the wall clock reads it directly (the zones
// run ~150,000 times a second, and QueryPerformanceCounter cost ~5% of the GPU thread).
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

// What the zones and drain waits measure in: the time stamp counter without the fences the wall
// clock reads it with (those fences cost ~10% of the GPU back thread with ~45,000 draws a
// second). Report converts it with the wall clock.
inline u64 Ticks() {
#ifdef ARCHITECTURE_x86_64
    return __rdtsc();
#else
    return Now();
#endif
}

/// TEST: amounts counted with EDEN_NX2_GPU_PROFILE=2, logged per second in a line of their own.
enum class FineCounter : u32 {
    UniformRebinds,  ///< Small uniform buffers bound again from the same stream allocation
    UniformStreamed, ///< Small uniform buffers copied into the stream buffer
    UniformCached,   ///< Uniform buffers bound from their host buffer
    UniformOther,    ///< Uniform buffers bound otherwise (root tables, OpenGL, fallbacks)
    UniformBytes,    ///< Bytes of the uniform buffers bound (all paths)
    StorageBinds,    ///< Graphics storage buffers bound
    TextureHandles,  ///< Graphics texture handles read
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(FineCounter::Count)>
    fine_counter_names{
        "uniform rebinds", "uniform streamed", "uniform cached",  "uniform other",
        "uniform bytes",   "storage binds",    "texture handles",
    };
static_assert(fine_counter_names.back() != nullptr, "a fine counter without a name");

struct ThreadData {
    std::array<u64, static_cast<size_t>(FineCounter::Count)> fine_counters{}; ///< TEST
    std::array<u64, static_cast<size_t>(Zone::Count)> exclusive_ns{};
    std::array<u64, static_cast<size_t>(Zone::Count)> count{};
    std::array<u64, 12> counters{}; ///< See Counter
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_ns{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_count{};
    u64 child_ns{};    ///< Time of the zones nested in the current one
    u64 window_start{};
    u64 window_start_ticks{};
    std::optional<u64> cpu_start_ns;
    u32 depth{};          ///< Zones open (Scope, timed SampledScope)
    u32 samples_left{1};  ///< SampledScope: entries until the next timed one
    u32 sample_weight{1}; ///< SampledScope: the entries the next timed one stands for
    u32 sample_seed{0x2545F491};
};

inline thread_local ThreadData t_data;

class Scope {
public:
    explicit Scope(Zone zone_) : Scope{zone_, Enabled()} {}

    /// A zone timed only if `on` (a profiled run's choice, such as Fine()).
    Scope(Zone zone_, bool on) : zone{zone_}, enabled{on} {
        if (!enabled) {
            return;
        }
        auto& data = t_data;
        ++data.depth;
        parent_child_ns = data.child_ns;
        data.child_ns = 0;
        start = Ticks();
    }

    ~Scope() {
        if (!enabled) {
            return;
        }
        const u64 elapsed = Ticks() - start;
        auto& data = t_data;
        const auto index = static_cast<size_t>(zone);
        data.exclusive_ns[index] += elapsed - data.child_ns;
        ++data.count[index];
        data.child_ns = parent_child_ns + elapsed;
        --data.depth;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Zone zone;
    bool enabled;
    u64 start{};
    u64 parent_child_ns{};
};

/// TEST: a fine zone (EDEN_NX2_GPU_PROFILE=2).
class FineScope : public Scope {
public:
    explicit FineScope(Zone zone_) : Scope{zone_, Fine()} {}
};

/// The ticks two time stamp reads in a row take: what a timed entry measures on top of its own
/// time (the least of a few tries).
inline u64 TimingOverheadTicks() {
    static const u64 overhead = [] {
        u64 least = ~u64{0};
        for (int i = 0; i < 64; ++i) {
            const u64 first = Ticks();
            const u64 second = Ticks();
            least = (std::min)(least, second - first);
        }
        return least;
    }();
    return overhead;
}

/// A zone entered millions of times a second, such as the split's back's 3D register writes
/// (Maxwell3D::CallMethod): timing every entry took two time stamp reads each, which cost more
/// than most entries and slowed the very runs it measured. Entered outside every other zone, one
/// entry in about SampleEvery is timed and stands for the entries since the last timed one; all
/// are counted. Entered inside another zone it does nothing: the entry is that zone's time.
class SampledScope {
public:
    static constexpr u32 SampleEvery = 32;
    /// About 25 us at 4 GHz (time stamp ticks): far longer than these entries take.
    static constexpr u64 LongTicks = 100'000;

    /// `entries`: what this entry counts as (a run of constant buffer words counts each word).
    explicit SampledScope(Zone zone, u32 entries = 1) {
        if (!Enabled()) [[likely]] {
            return;
        }
        auto& data = t_data;
        if (data.depth != 0) {
            return;
        }
        index = static_cast<size_t>(zone);
        data.count[index] += entries;
        if (--data.samples_left != 0) {
            return;
        }
        weight = data.sample_weight;
        // The next interval: 1 to 2 * SampleEvery - 1 entries, so that no fixed period lines up
        // with a repeating pattern of entries (xorshift32).
        u32 seed = data.sample_seed;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        data.sample_seed = seed;
        data.sample_weight = 1 + seed % (2 * SampleEvery - 1);
        data.samples_left = data.sample_weight;
        ++data.depth;
        parent_child_ns = data.child_ns;
        data.child_ns = 0;
        start = Ticks();
    }

    ~SampledScope() {
        if (weight == 0) {
            return;
        }
        const u64 elapsed = Ticks() - start;
        auto& data = t_data;
        // Less the reads' own cost, which the untimed entries did not pay.
        const u64 overhead = TimingOverheadTicks();
        const u64 own = elapsed - data.child_ns;
        const u64 measured = own > overhead ? own - overhead : 0;
        // An entry this long was most likely interrupted or preempted: it counts once, not for
        // the untimed entries too.
        data.exclusive_ns[index] += measured > LongTicks ? measured : measured * weight;
        data.child_ns = parent_child_ns + elapsed;
        --data.depth;
    }

    SampledScope(const SampledScope&) = delete;
    SampledScope& operator=(const SampledScope&) = delete;

private:
    size_t index{};
    u64 weight{}; ///< 0: not timed
    u64 start{};
    u64 parent_child_ns{};
};

/// Amounts counted on the GPU thread, logged per second with the zones.
enum class Counter : u32 {
    DownloadBytes,  ///< Buffer data copied back from the host GPU at fences
    DownloadCopies, ///< Buffer ranges copied back
    DmaReadBytes,   ///< Guest memory the native MME read with DMA
    DownloadBytesOnDmaPages, ///< Of DownloadBytes: on pages the MME has DMA-read
    MethodWrites,   ///< 3D register writes (Maxwell3D::CallMethod below the macro range)
    SameValueWrites, ///< Of MethodWrites: the register already held the value
    AcquireChecks,  ///< Semaphore acquires evaluated
    AcquireWaits,   ///< Of AcquireChecks: not met yet (the channel waits)
    /// TEST: the split's front publishes words while its back waits for work (GpuSplit::Publish):
    /// the back's wait since it ran out or since the first of them was sent (Ticks), and count
    LatePublishTicks,
    LatePublishes,
    ResnapshotWords, ///< TEST: command list words copied again at resumes
    EarlyPublishes,  ///< TEST: publishes to a back waiting for work, before publish_every
};

/// Method calls on this thread by method (3D registers on the back, macros on the front), and
/// same-value writes among them; MME DMA reads by the macro that made them. The most frequent
/// are logged with the zones.
inline thread_local std::array<u32, 0x2000> t_method_count{};
inline thread_local std::array<u32, 0x2000> t_method_same{};
inline thread_local std::array<u32, 0x2000> t_dma_reads{};

inline void CountMethod(u32 method, bool same) {
    if (Enabled() && method < t_method_count.size()) {
        ++t_method_count[method];
        t_method_same[method] += same ? 1 : 0;
    }
}

/// A 3D register write (Maxwell3D::CallMethod below the macro range), `same` if the register
/// held the value already: the write counters and the per-method counts.
inline void CountRegisterWrite(u32 method, bool same) {
    if (!Enabled()) {
        return;
    }
    auto& counters = t_data.counters;
    ++counters[static_cast<size_t>(Counter::MethodWrites)];
    counters[static_cast<size_t>(Counter::SameValueWrites)] += same ? 1 : 0;
    CountMethod(method, same);
}

inline void CountDmaRead(u32 macro) {
    if (Enabled() && macro < t_dma_reads.size()) {
        ++t_dma_reads[macro];
    }
}

inline std::string TopMethods(const std::array<u32, 0x2000>& counts,
                              const std::array<u32, 0x2000>* same, u64 window) {
    std::array<u32, 0x2000> order;
    for (u32 i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    constexpr size_t top = 24;
    std::partial_sort(order.begin(), order.begin() + top, order.end(),
                      [&counts](u32 a, u32 b) { return counts[a] > counts[b]; });
    std::string text;
    for (size_t i = 0; i < top && counts[order[i]] != 0; ++i) {
        const u32 method = order[i];
        text += fmt::format(" {:#x} {:.0f}/s", method,
                            static_cast<double>(counts[method]) * 1e9 / window);
        if (same != nullptr && (*same)[method] != 0) {
            text += fmt::format(" ({:.0f} same)", static_cast<double>((*same)[method]) * 1e9 / window);
        }
        text += ',';
    }
    if (!text.empty()) {
        text.pop_back();
    }
    return text;
}

/// Device pages (4 KiB) the native MME has DMA-read since start (GPU thread only; filled only
/// while profiling): which buffer downloads anything on the GPU thread reads back.
inline std::unordered_set<u64>& DmaReadPages() {
    static std::unordered_set<u64> pages;
    return pages;
}

inline void Add(Counter counter, u64 amount) {
    if (Enabled()) {
        t_data.counters[static_cast<size_t>(counter)] += amount;
    }
}

/// TEST: EDEN_NX2_GPU_PROFILE=2 only.
inline void AddFine(FineCounter counter, u64 amount) {
    if (Fine()) [[unlikely]] {
        t_data.fine_counters[static_cast<size_t>(counter)] += amount;
    }
}

/// The front waited for the back since `start` (Ticks) for `site`.
inline void AddDrain(DrainSite site, u64 start) {
    if (Enabled()) {
        const auto index = static_cast<size_t>(site);
        t_data.drain_ns[index] += Ticks() - start;
        ++t_data.drain_count[index];
    }
}

/// Called by the GPU thread (and the split's back thread) between commands: logs and resets
/// the totals every 10 s.
inline void Report(const char* thread_name = "GPU thread") {
    if (!Enabled()) {
        return;
    }
    auto& data = t_data;
    const u64 now = Now();
    if (data.window_start == 0) {
        data.window_start = now;
        data.window_start_ticks = Ticks();
        data.cpu_start_ns = Common::CurrentThreadCpuTimeNs();
        data.child_ns = 0;
        return;
    }
    const u64 window = now - data.window_start;
    if (window < 10'000'000'000ULL) {
        return;
    }
    const auto cpu_now = Common::CurrentThreadCpuTimeNs();
    const u64 now_ticks = Ticks();
    const u64 window_ticks = (std::max<u64>)(now_ticks - data.window_start_ticks, 1);
    // Time in no zone at all is the wall time minus the zones' totals.
    u64 zoned{};
    for (size_t i = 1; i < data.exclusive_ns.size(); ++i) {
        zoned += data.exclusive_ns[i];
    }
    data.exclusive_ns[0] = window_ticks > zoned ? window_ticks - zoned : 0;
    std::string line;
    for (size_t i = 0; i < data.exclusive_ns.size(); ++i) {
        if (data.exclusive_ns[i] == 0 && data.count[i] == 0) {
            continue;
        }
        // Per second of wall time: milliseconds spent and calls made.
        const double ms = static_cast<double>(data.exclusive_ns[i]) * 1000.0 / window_ticks;
        const double calls = static_cast<double>(data.count[i]) * 1e9 / window;
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s)", zone_names[i], ms, calls);
    }
    line += fmt::format(" | buffer downloads {:.0f} KiB in {:.0f} copies ({:.0f} KiB on pages "
                        "MME DMA reads) | MME DMA reads {:.0f} KiB",
                        static_cast<double>(data.counters[0]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[1]) * 1e9 / window,
                        static_cast<double>(data.counters[3]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[2]) * 1e9 / 1024.0 / window);
    line += fmt::format(" | 3D register writes {:.0f} ({:.0f} same value)",
                        static_cast<double>(data.counters[4]) * 1e9 / window,
                        static_cast<double>(data.counters[5]) * 1e9 / window);
    line += fmt::format(" | acquires {:.0f} ({:.0f} waited)",
                        static_cast<double>(data.counters[6]) * 1e9 / window,
                        static_cast<double>(data.counters[7]) * 1e9 / window);
    if (data.counters[9] != 0 || data.counters[10] != 0 || data.counters[11] != 0) {
        // TEST: the back idle while words it could run waited on the front.
        line += fmt::format(" | publishes to an idle back late by {:.1f} ms ({:.0f}/s), early "
                            "{:.0f}/s | resnapshot words {:.0f}/s",
                            static_cast<double>(data.counters[8]) * 1000.0 / window_ticks,
                            static_cast<double>(data.counters[9]) * 1e9 / window,
                            static_cast<double>(data.counters[11]) * 1e9 / window,
                            static_cast<double>(data.counters[10]) * 1e9 / window);
    }
    std::string drains;
    for (size_t i = 0; i < data.drain_count.size(); ++i) {
        if (data.drain_count[i] != 0) {
            drains += fmt::format(" {} {:.1f} ms ({:.0f}/s),", drain_site_names[i],
                                  static_cast<double>(data.drain_ns[i]) * 1000.0 / window_ticks,
                                  static_cast<double>(data.drain_count[i]) * 1e9 / window);
        }
    }
    if (!drains.empty()) {
        drains.pop_back();
        line += " | waits for the back by site:" + drains;
    }
    if (cpu_now && data.cpu_start_ns && *cpu_now >= *data.cpu_start_ns) {
        line += fmt::format(" | host_cpu_busy_ms_per_s={:.3f} window_wall_ms={:.3f}",
                            static_cast<double>(*cpu_now - *data.cpu_start_ns) * 1000.0 / window,
                            static_cast<double>(window) / 1e6);
    } else {
        line += fmt::format(" | host_cpu_busy_ms_per_s=unavailable window_wall_ms={:.3f}",
                            static_cast<double>(window) / 1e6);
    }
    LOG_INFO(HW_GPU, "{} per second:{}", thread_name, line);
    if (Fine()) {
        // TEST: the fine counters.
        std::string fine;
        for (size_t i = 0; i < data.fine_counters.size(); ++i) {
            if (data.fine_counters[i] != 0) {
                fine += fmt::format(" {} {:.0f},", fine_counter_names[i],
                                    static_cast<double>(data.fine_counters[i]) * 1e9 / window);
            }
        }
        if (!fine.empty()) {
            fine.pop_back();
            LOG_INFO(HW_GPU, "{} fine counters per second:{}", thread_name, fine);
        }
    }
    if (const std::string methods = TopMethods(t_method_count, &t_method_same, window); !methods.empty()) {
        LOG_INFO(HW_GPU, "{} methods per second:{}", thread_name, methods);
    }
    if (const std::string reads = TopMethods(t_dma_reads, nullptr, window); !reads.empty()) {
        LOG_INFO(HW_GPU, "{} MME DMA reads per second by macro:{}", thread_name, reads);
    }
    t_method_count = {};
    t_method_same = {};
    t_dma_reads = {};
    const u32 depth = data.depth;
    data = {};
    data.depth = depth;
    data.window_start = now;
    data.window_start_ticks = now_ticks;
    data.cpu_start_ns = cpu_now;
}

} // namespace VideoCommon::Nx2Profile
