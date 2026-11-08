// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"

// EDEN_NX2_DRAIN_CHECK: what the split GPU thread's front would see without some of its waits for
// the back, counted without changing what runs and logged every 10 s by the front:
// - command lists on pages no GPU work had written that did not wait for the compute and copy
//   work sent after the last host-level wait (DmaPusher::Step): how many of them that work made
//   GPU-written, checked on the back at the list's place. It must stay 0;
// - semaphore acquires, and the polls of suspended ones (Puller): whether the value met the
//   acquire before the front drained the split and after, by whether the GPU had released a
//   semaphore at that address before (any engine, RasterizerInterface::Query with a payload);
// - acquires and polls the semaphore view decides (Puller AskView), verified: the front drains
//   and reads the word as before anyway, and counts the decisions the drain contradicts. A pass
//   the drained word does not meet must stay 0; such a pass is corrected (the channel waits), and
//   a wait the drained word meets ends (counted: a drain would have passed or resumed it);
// - the releases the front noted for the QMD launches it valued (FrontQmdView), against the ones
//   the back then ran, compared on the back right after the QMD methods, and the descriptors the
//   front read for them (from memory, the signal's inline QMD or a queued one) against the ones
//   the back read; the descriptors the front read for signals it valued late (they waited) against
//   the ones the back read running them, by their position; whether the back could have run an
//   entry the front valued when it read memory for it. Every 512th time, and after every signal
//   valued late, the front's whole QMD table against the back's. Wrong ones must stay 0.
namespace VideoCommon::Nx2DrainCheck {

// A plain flag: the front checks it at every command list and acquire.
inline bool g_enabled = std::getenv("EDEN_NX2_DRAIN_CHECK") != nullptr;

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_enabled = std::getenv("EDEN_NX2_DRAIN_CHECK") != nullptr;
}

struct State {
    std::atomic<u64> lists_unwaited{};
    std::atomic<u64> lists_written{};
    /// Indexed by poll * 8 + released * 4 + met before the drain * 2 + met after it.
    std::array<std::atomic<u64>, 16> acquires{};
    /// Passes the semaphore view decided, by poll * 3 + ViewPass, and those the drained word did
    /// not meet.
    std::array<std::atomic<u64>, 6> view_passes{};
    std::array<std::atomic<u64>, 6> view_wrong{};
    /// Passes the drained word didn't meet because a fenced release landed during the drain: the
    /// drain path itself may read either value then.
    std::array<std::atomic<u64>, 6> view_raced{};
    /// Acquires the view suspended and polls it kept waiting (by poll), and of those the ones
    /// the drained word met.
    std::array<std::atomic<u64>, 2> view_waits{};
    std::array<std::atomic<u64>, 2> view_waits_met{};
    /// QMD methods the front valued and the releases compared for them, those that differed,
    /// and methods it could not value (their releases are not compared).
    std::atomic<u64> qmd_methods{};
    std::atomic<u64> qmd_releases{};
    std::atomic<u64> qmd_wrong{};
    std::atomic<u64> qmd_unvalued{};
    std::atomic<u32> qmd_logged{};
    /// Descriptors compared for the valued methods, and those the back read otherwise; QMD tables
    /// compared, and those that differed.
    std::atomic<u64> qmd_reads{};
    std::atomic<u64> qmd_reads_wrong{};
    std::atomic<u32> qmd_reads_logged{};
    std::atomic<u64> qmd_tables{};
    std::atomic<u64> qmd_tables_wrong{};
    /// Of the tables, those compared right after the front valued signals late; signals valued
    /// late whose descriptors were compared, those the back read otherwise, and those whose back
    /// reads were gone (more than LateReadsKept waited); QMD methods the front valued after the
    /// back could have run them (their entry published).
    std::atomic<u64> qmd_tables_late{};
    std::atomic<u64> qmd_late_signals{};
    std::atomic<u64> qmd_late_wrong{};
    std::atomic<u64> qmd_late_missing{};
    std::atomic<u64> qmd_published_early{};
    std::mutex released_mutex;
    std::unordered_set<u64> released; ///< Device addresses of the semaphores the GPU released
    std::chrono::steady_clock::time_point last_report{}; ///< The front's
};

inline State& Get() {
    static State state;
    return state;
}

/// The back runs a semaphore release at `device_address`.
inline void NoteRelease(u64 device_address) {
    State& state = Get();
    std::scoped_lock lock{state.released_mutex};
    state.released.insert(device_address & ~u64{3});
}

[[nodiscard]] inline bool WasReleased(u64 device_address) {
    State& state = Get();
    std::scoped_lock lock{state.released_mutex};
    return state.released.contains(device_address & ~u64{3});
}

/// An acquire, or the poll of a suspended one (`poll`), at a semaphore the GPU had `released`:
/// whether the value met it before the drain and after.
inline void NoteAcquire(bool poll, bool released, bool met_before, bool met_after) {
    const size_t index =
        (poll ? 8U : 0U) + (released ? 4U : 0U) + (met_before ? 2U : 0U) + (met_after ? 1U : 0U);
    Get().acquires[index].fetch_add(1, std::memory_order_relaxed);
}

/// How the semaphore view passed an acquire or resumed a poll.
enum class ViewPass : u32 {
    Memory, ///< Met in memory
    Queued, ///< Met by the last release sent to the word, still queued
    Waited, ///< Met after a positioned wait (GpuSplit::WaitForPosition)
};

/// The view passed an acquire (or resumed a poll, `poll`); whether the word read after a drain
/// met it too, and if not, whether a fenced release that landed during the drain changed it
/// (`raced`).
inline void NoteViewPass(bool poll, ViewPass how, bool met_after_drain, bool raced) {
    const size_t index = (poll ? 3U : 0U) + static_cast<size_t>(how);
    State& state = Get();
    state.view_passes[index].fetch_add(1, std::memory_order_relaxed);
    if (!met_after_drain) {
        (raced ? state.view_raced : state.view_wrong)[index].fetch_add(1,
                                                                       std::memory_order_relaxed);
    }
}

/// The view suspended an acquire (or kept a poll waiting, `poll`); whether the word read after
/// a drain met it.
inline void NoteViewWait(bool poll, bool met_after_drain) {
    State& state = Get();
    state.view_waits[poll ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
    if (met_after_drain) {
        state.view_waits_met[poll ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
    }
}

/// A release of a QMD launch: the word and what it holds once the back ran it, or a value the
/// front could not tell (`known` false), or that value at least (`at_least`).
struct QmdRelease {
    u64 device;
    u32 value;
    bool known;
    bool at_least;

    bool operator==(const QmdRelease&) const = default;
};

/// A QMD the scheduling read (NativeQmdScheduler::Signal's read): its GPU address and words.
struct QmdRead {
    u64 address;
    std::array<u32, 64> words;
};

/// The releases the back ran for QMD launches since the last check (the back's thread only).
inline std::vector<QmdRelease>& BackQmdReleases() {
    static std::vector<QmdRelease> released;
    return released;
}

/// The QMDs the back's scheduling read since the last check (the back's thread only).
inline std::vector<QmdRead>& BackQmdReads() {
    static std::vector<QmdRead> reads;
    return reads;
}

/// Where one signal's reads end in BackQmdReads (no QMD lives at that address).
inline constexpr u64 SignalEnd = ~u64{0};

/// The descriptors the back read for the signals that waited on the front, by the position after
/// their entry and their place in it, until the front valued them late (the back's thread only).
inline std::map<std::pair<u64, u32>, std::vector<QmdRead>>& LateBackReads() {
    static std::map<std::pair<u64, u32>, std::vector<QmdRead>> reads;
    return reads;
}
inline constexpr size_t LateReadsKept = 16384;

/// A signal the front valued late (it waited), at the entry ending at `position`, `index`-th of
/// the entry's signals, and the descriptors it read for it.
struct LateQmdRead {
    u64 position;
    u32 index;
    std::vector<QmdRead> reads;
};

/// The split's back's scheduling ran one signal (KeplerCompute::ProcessNativeMethod).
inline void NoteQmdSignalEnd(bool on_back) {
    if (on_back) {
        BackQmdReads().push_back({SignalEnd, {}});
    }
}

/// The front valued a QMD method after its entry was published: the back could have run it.
inline void NotePublishedEarly() {
    State& state = Get();
    state.qmd_published_early.fetch_add(1, std::memory_order_relaxed);
    if (state.qmd_reads_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(HW_GPU, "QMD view check: the front valued a QMD method the back could run");
    }
}

/// The split's back runs a release of a QMD launch (KeplerCompute::ProcessNativeMethod).
inline void NoteQmdRelease(u64 device_address, u32 value, bool on_back) {
    if (on_back) {
        BackQmdReleases().push_back({device_address, value, true, false});
    }
}

/// The split's back's scheduling read a QMD (KeplerCompute::ProcessNativeMethod).
inline void NoteQmdRead(u64 address, const std::array<u32, 64>& words, bool on_back) {
    if (on_back) {
        BackQmdReads().push_back({address, words});
    }
}

/// The back compared the front's QMD table with its own at the same place (`same`); `late`: right
/// after the front valued signals late.
inline void NoteQmdTables(bool same, size_t states, bool late = false) {
    State& state = Get();
    state.qmd_tables.fetch_add(1, std::memory_order_relaxed);
    if (late) {
        state.qmd_tables_late.fetch_add(1, std::memory_order_relaxed);
    }
    if (!same) {
        state.qmd_tables_wrong.fetch_add(1, std::memory_order_relaxed);
        if (state.qmd_reads_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(HW_GPU,
                      "QMD view check: the front's table ({} QMDs) differs from the back's{}",
                      states, late ? " after it valued signals late" : "");
        }
    }
}

/// The back, after the signals the front valued late: the descriptors it read for them against
/// the ones it read itself running them.
inline void CheckLateQmdReads(std::span<const LateQmdRead> late) {
    State& state = Get();
    auto& kept = LateBackReads();
    for (const LateQmdRead& signal : late) {
        const auto it = kept.find({signal.position, signal.index});
        if (it == kept.end()) {
            state.qmd_late_missing.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        state.qmd_late_signals.fetch_add(1, std::memory_order_relaxed);
        const auto& back_reads = it->second;
        bool same = back_reads.size() == signal.reads.size();
        for (size_t i = 0; same && i < back_reads.size(); ++i) {
            same = back_reads[i].address == signal.reads[i].address &&
                   back_reads[i].words == signal.reads[i].words;
        }
        if (!same) {
            state.qmd_late_wrong.fetch_add(1, std::memory_order_relaxed);
            if (state.qmd_reads_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                std::string text;
                for (size_t i = 0; i < (std::max)(back_reads.size(), signal.reads.size()); ++i) {
                    const QmdRead* back = i < back_reads.size() ? &back_reads[i] : nullptr;
                    const QmdRead* front = i < signal.reads.size() ? &signal.reads[i] : nullptr;
                    text += fmt::format(" | {:#x} by the back, {:#x} by the front",
                                        back ? back->address : 0, front ? front->address : 0);
                    for (size_t word = 0; back && front && word < 64; ++word) {
                        if (back->words[word] != front->words[word]) {
                            text += fmt::format(", word {} {:#x} for {:#x}", word,
                                                back->words[word], front->words[word]);
                            break;
                        }
                    }
                }
                LOG_ERROR(HW_GPU,
                          "QMD view check: the back read other descriptors for a signal the "
                          "front valued late (position {:#x}, signal {}){}",
                          signal.position, signal.index, text);
            }
        }
        kept.erase(it);
    }
}

/// The back, right after QMD methods the front sent: what it read and released for them against
/// what the front read and noted (`front_reads`, `predicted`, in order), or nothing to compare
/// where the front could not value them (`valued` false).
/// `wait_position`: the signals of the entry ending there waited on the front from the
/// `first_waiting`-th on (0: none): their reads are kept for CheckLateQmdReads.
inline void CheckQmdReleases(std::span<const QmdRelease> predicted,
                             std::span<const QmdRead> front_reads, bool valued,
                             u64 wait_position = 0, u32 first_waiting = 0) {
    State& state = Get();
    auto& released = BackQmdReleases();
    auto& all_reads = BackQmdReads();
    // By signal: each ends with a SignalEnd.
    std::vector<QmdRead> back_reads;
    std::vector<std::vector<QmdRead>> signals(1);
    size_t ends = 0;
    for (QmdRead& read : all_reads) {
        if (read.address == SignalEnd) {
            signals.emplace_back();
            ++ends;
            continue;
        }
        back_reads.push_back(read);
        signals.back().push_back(std::move(read));
    }
    signals.resize(ends);
    all_reads.clear();
    if (!valued) {
        state.qmd_unvalued.fetch_add(1, std::memory_order_relaxed);
        released.clear();
        auto& kept = LateBackReads();
        if (wait_position != 0) {
            // Kept from before the split last started over at position 0.
            kept.erase(kept.lower_bound({wait_position, 0}), kept.end());
        }
        for (u32 index = first_waiting; wait_position != 0 && index < signals.size(); ++index) {
            if (kept.size() >= LateReadsKept) {
                kept.erase(kept.begin()); // The oldest: the front lost or dropped it
            }
            kept.insert_or_assign({wait_position, index}, std::move(signals[index]));
        }
        return;
    }
    state.qmd_reads.fetch_add(back_reads.size(), std::memory_order_relaxed);
    bool same_reads = back_reads.size() == front_reads.size();
    for (size_t i = 0; same_reads && i < back_reads.size(); ++i) {
        same_reads = back_reads[i].address == front_reads[i].address &&
                     back_reads[i].words == front_reads[i].words;
    }
    if (!same_reads) {
        state.qmd_reads_wrong.fetch_add(1, std::memory_order_relaxed);
        if (state.qmd_reads_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            std::string text;
            for (size_t i = 0; i < (std::max)(back_reads.size(), front_reads.size()); ++i) {
                const QmdRead* back = i < back_reads.size() ? &back_reads[i] : nullptr;
                const QmdRead* front = i < front_reads.size() ? &front_reads[i] : nullptr;
                text += fmt::format(" | {:#x} by the back, {:#x} by the front",
                                    back ? back->address : 0, front ? front->address : 0);
                for (size_t word = 0; back && front && word < 64; ++word) {
                    if (back->words[word] != front->words[word]) {
                        text += fmt::format(", word {} {:#x} for {:#x}", word, back->words[word],
                                            front->words[word]);
                        break;
                    }
                }
            }
            LOG_ERROR(HW_GPU, "QMD view check: the back read other descriptors{}", text);
        }
    }
    state.qmd_methods.fetch_add(1, std::memory_order_relaxed);
    state.qmd_releases.fetch_add(released.size(), std::memory_order_relaxed);
    bool same = released.size() == predicted.size();
    for (size_t i = 0; same && i < released.size(); ++i) {
        const QmdRelease& note = predicted[i];
        same = released[i].device == note.device &&
               (!note.known || (note.at_least ? released[i].value >= note.value
                                              : released[i].value == note.value));
    }
    if (!same) {
        state.qmd_wrong.fetch_add(1, std::memory_order_relaxed);
        if (state.qmd_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            std::string text;
            for (const QmdRelease& release : released) {
                text += fmt::format(" {:#x}={:#x}", release.device, release.value);
            }
            text += " | noted";
            for (const QmdRelease& note : predicted) {
                text += fmt::format(" {:#x}{}{:#x}", note.device,
                                    !note.known     ? "=?"
                                    : note.at_least ? ">="
                                                    : "=",
                                    note.value);
            }
            LOG_ERROR(HW_GPU, "QMD view check: the back released{}", text);
        }
    }
    released.clear();
}

/// The front fetches a list without waiting for writers still queued after the last sync.
inline void NoteListUnwaited() {
    Get().lists_unwaited.fetch_add(1, std::memory_order_relaxed);
}

/// The back reached that list: whether the work before it made the list's pages GPU-written.
inline void NoteListChecked(bool written) {
    if (written) {
        Get().lists_written.fetch_add(1, std::memory_order_relaxed);
    }
}

/// The front, between commands.
inline void MaybeReport() {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    const auto now = std::chrono::steady_clock::now();
    if (state.last_report == std::chrono::steady_clock::time_point{}) {
        state.last_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - state.last_report).count();
    if (seconds < 10.0) {
        return;
    }
    state.last_report = now;
    const auto take = [](std::atomic<u64>& value) {
        return value.exchange(0, std::memory_order_relaxed);
    };
    const auto group = [&](bool poll, bool released) {
        const size_t base = (poll ? 8U : 0U) + (released ? 4U : 0U);
        const u64 neither = take(state.acquires[base]);
        const u64 only_after = take(state.acquires[base + 1]);
        const u64 only_before = take(state.acquires[base + 2]);
        const u64 both = take(state.acquires[base + 3]);
        return fmt::format("{} (met before and after the drain {}, only after {}, only before {}, "
                           "neither {})",
                           neither + only_after + only_before + both, both, only_after, only_before,
                           neither);
    };
    const auto passes = [&](bool poll) {
        const size_t base = poll ? 3U : 0U;
        const u64 by_memory = take(state.view_passes[base]);
        const u64 by_queued = take(state.view_passes[base + 1]);
        const u64 waited = take(state.view_passes[base + 2]);
        const u64 wrong_memory = take(state.view_wrong[base]);
        const u64 wrong_queued = take(state.view_wrong[base + 1]);
        const u64 wrong_waited = take(state.view_wrong[base + 2]);
        const u64 raced = take(state.view_raced[base]) + take(state.view_raced[base + 1]) +
                          take(state.view_raced[base + 2]);
        return fmt::format("{} (by memory {}, by a release still queued {}, after a positioned "
                           "wait {}), not met after the drain {} (of those by memory {}, queued "
                           "{}, after a wait {}), changed by a fenced release landing during the "
                           "drain {}",
                           by_memory + by_queued + waited, by_memory, by_queued, waited,
                           wrong_memory + wrong_queued + wrong_waited, wrong_memory, wrong_queued,
                           wrong_waited, raced);
    };
    const auto waits = [&](bool poll) {
        const u64 count = take(state.view_waits[poll ? 1 : 0]);
        const u64 met = take(state.view_waits_met[poll ? 1 : 0]);
        return fmt::format("{} (met after the drain {})", count, met);
    };
    const u64 unwaited = take(state.lists_unwaited);
    const u64 written = take(state.lists_written);
    const std::string acquires_never = group(false, false);
    const std::string acquires_released = group(false, true);
    const std::string polls_never = group(true, false);
    const std::string polls_released = group(true, true);
    const std::string acquire_passes = passes(false);
    const std::string acquire_waits = waits(false);
    const std::string poll_passes = passes(true);
    const std::string poll_waits = waits(true);
    const u64 qmd_methods = take(state.qmd_methods);
    const u64 qmd_releases = take(state.qmd_releases);
    const u64 qmd_wrong = take(state.qmd_wrong);
    const u64 qmd_unvalued = take(state.qmd_unvalued);
    const u64 qmd_reads = take(state.qmd_reads);
    const u64 qmd_reads_wrong = take(state.qmd_reads_wrong);
    const u64 qmd_tables = take(state.qmd_tables);
    const u64 qmd_tables_wrong = take(state.qmd_tables_wrong);
    const u64 qmd_tables_late = take(state.qmd_tables_late);
    const u64 qmd_late_signals = take(state.qmd_late_signals);
    const u64 qmd_late_wrong = take(state.qmd_late_wrong);
    const u64 qmd_late_missing = take(state.qmd_late_missing);
    const u64 qmd_published_early = take(state.qmd_published_early);
    LOG_INFO(HW_GPU,
             "Drain checks in {:.1f} s: command lists that did not wait for writers sent after the "
             "last sync {}, GPU-written by those writers {} | drained acquires at addresses the "
             "GPU never released {}, at ones it released {} | drained polls of suspended acquires "
             "at addresses never released {}, at released ones {} | semaphore view: acquires "
             "passed {}, suspended {}; polls resumed {}, kept waiting {} | QMD view: methods "
             "valued {} with {} releases, released otherwise than noted {}, with {} descriptors "
             "read, read otherwise {}; valued after the back could run them {}; signals valued "
             "late {}, read otherwise {}, not compared {}; tables compared {} (after valuing late "
             "{}), differing {}; not valued {}",
             seconds, unwaited, written, acquires_never, acquires_released, polls_never,
             polls_released, acquire_passes, acquire_waits, poll_passes, poll_waits, qmd_methods,
             qmd_releases, qmd_wrong, qmd_reads, qmd_reads_wrong, qmd_published_early,
             qmd_late_signals, qmd_late_wrong, qmd_late_missing, qmd_tables, qmd_tables_late,
             qmd_tables_wrong, qmd_unvalued);
}

} // namespace VideoCommon::Nx2DrainCheck

// EDEN_NX2_ACQUIRE_STATS: how the split's front decides Switch 2 semaphore acquires (Puller), every
// 10 s, without changing what runs: by operation, which of them still wait, at which words (on
// GPU-written pages, by the releases the front sent there), what resumes them and how long they
// waited. Front thread only, but for the writer that latched a waiter (Nx2SemaphoreWaiters).
namespace VideoCommon::Nx2AcquireStats {

// A plain flag: the front checks it at every acquire and release it sends.
inline bool g_enabled = std::getenv("EDEN_NX2_ACQUIRE_STATS") != nullptr;

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_enabled = std::getenv("EDEN_NX2_ACQUIRE_STATS") != nullptr;
}

/// How the front decided an acquire.
enum class Decision : u32 {
    Memory,    ///< Passed: met in memory
    Queued,    ///< Passed: met by the last release sent to the word, still queued
    Waited,    ///< Passed after a positioned wait
    Suspended, ///< The channel waits
    Drained,   ///< The semaphore view doesn't apply: drained as before
    Count,
};

/// Who the front sent releases to a word for (State::noted bits).
enum class Releaser : u32 {
    Puller,        ///< Semaphore methods (WriteLong, SEM_EXECUTE release)
    Report3D,      ///< 3D report semaphores
    ReportCompute, ///< Compute report semaphores
    Copy,          ///< Copy engine semaphores
    Qmd,           ///< QMD launches the front valued (FrontQmdView)
    Count,
};

/// Why the front's QMD view could not value a signal (FrontQmdView::Signal), or one that waited.
enum class QmdLost : u32 {
    NotSynced,        ///< Its table is not the back's yet (the back has not caught up since)
    Unplaced,         ///< Work that may write guest memory anywhere was still queued
    GpuWritten,       ///< A QMD it read is on a page GPU work wrote
    Pending,          ///< A write into a QMD it read was still queued
    ReleasedInto,     ///< A release of the same signal writes into a QMD it read
    Mappings,         ///< Work that may change the GPU's mappings was still queued
    WrittenAfter,     ///< Waiting, a QMD it reads was written after it
    MappedAfter,      ///< Waiting, the mappings may have changed after it
    LateReleasedInto, ///< Waiting, a release of it writes a QMD read late
    TooMany,          ///< Too many signals waited
    Count,
};

/// Where the front's QMD view took the back's table over (FrontQmdView::TakeOver).
enum class QmdTakeOver : u32 {
    FrontStart,      ///< An engine's front started, with the back drained
    BetweenCommands, ///< The back had run everything sent (GPU thread loop)
    QmdMethod,       ///< The same, at a QMD method
    Count,
};

/// What the QMD view noted for a release.
enum class QmdNote : u32 {
    Exact,
    AtLeast, ///< A maximum with a word it can't tell
    Unknown, ///< A reduction of a word it can't tell
    Count,
};

/// What resumed a suspended acquire.
enum class Resume : u32 {
    Latch,  ///< A release the back ran latched the waiter (by its Source)
    Memory, ///< Met in memory: the CPU, or a GPU write that latches no waiter
    Queued, ///< Met by the last release sent to the word, still queued
    Drain,  ///< EDEN_NX2_DRAIN_CHECK's drain met it first
    Count,
};

using Source = Tegra::Nx2SemaphoreWaiters::Source;

struct State {
    /// By operation (equal, greater or equal, mask) and Decision.
    std::array<std::array<u64, static_cast<size_t>(Decision::Count)>, 3> acquires{};
    u64 positioned_waits{};
    u64 suspended_gpu_written{}; ///< Words on GPU-written pages
    u64 suspended_flush{};       ///< Of those, words the buffer cache held newer than memory
    std::array<u64, static_cast<size_t>(Releaser::Count)> suspended_noted{};
    u64 suspended_never_noted{}; ///< No release the front sent: QMD releases, the CPU
    std::array<u64, static_cast<size_t>(Resume::Count)> resumes{};
    std::array<u64, static_cast<size_t>(Source::Count)> latches{};
    u64 resumes_after_flush{};
    u64 wait_ns{};
    u64 wait_max_ns{};
    u64 waits_over_1ms{};
    u64 passed_by_qmd{}; ///< Of the passes by a queued release, at words QMD launches release
    u64 qmd_valued{};
    std::array<u64, static_cast<size_t>(QmdLost::Count)> qmd_lost{};
    /// Of qmd_lost[Unplaced] and qmd_lost[Mappings]: by what sent the last such write
    std::array<u64, static_cast<size_t>(UnplacedKind::Count)> qmd_lost_unplaced{};
    std::array<u64, static_cast<size_t>(UnplacedKind::Count)> qmd_lost_mappings{};
    std::array<u64, static_cast<size_t>(QmdNote::Count)> qmd_notes{};
    u64 qmd_inline_reads{}; ///< Descriptors read from an inline QMD still queued
    /// Signals that waited first, by why, and behind one; valued late, and how often the last
    /// waiting one was; waits that lost the table, by why, and the waiting signals they dropped;
    /// waiting signals takeovers dropped (the back had run them).
    std::array<u64, static_cast<size_t>(QmdLost::Count)> qmd_waits{};
    u64 qmd_behind{};
    u64 qmd_late{};
    u64 qmd_caught_up{};
    std::array<u64, static_cast<size_t>(QmdLost::Count)> qmd_late_lost{};
    u64 qmd_late_dropped{};
    u64 qmd_takeover_dropped{};
    /// Of qmd_waits[Pending]: by what wrote the QMD (FrontQmdView::NoteWaitWriter)
    std::map<std::string, u64> qmd_wait_writers;
    std::array<u64, static_cast<size_t>(QmdTakeOver::Count)> qmd_syncs{};
    std::chrono::steady_clock::duration qmd_sync_time{};
    std::unordered_map<u64, u8> noted; ///< Releaser bits by word
    std::chrono::steady_clock::time_point last_report{};
};

inline State& Get() {
    static State state;
    return state;
}

[[nodiscard]] inline size_t OperationIndex(u32 operation) {
    switch (operation) {
    case Tegra::Nx2SemaphoreWaiters::AcquireEqual:
        return 0;
    case Tegra::Nx2SemaphoreWaiters::AcquireGequal:
        return 1;
    default:
        return 2;
    }
}

/// The front sent a release to the word at `device_address`.
inline void NoteRelease(u64 device_address, Releaser releaser) {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    if (state.noted.size() < 65536) {
        state.noted[device_address & ~u64{3}] |= static_cast<u8>(1U << static_cast<u32>(releaser));
    }
}

/// The QMD view valued a signal (`lost` empty) or not; `kind`: what sent the last write that
/// may write anywhere (Unplaced) or change the mappings (Mappings).
inline void NoteQmdSignal(std::optional<QmdLost> lost, UnplacedKind kind = UnplacedKind::Other) {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    if (!lost) {
        ++state.qmd_valued;
        return;
    }
    ++state.qmd_lost[static_cast<size_t>(*lost)];
    if (*lost == QmdLost::Unplaced) {
        ++state.qmd_lost_unplaced[static_cast<size_t>(kind)];
    } else if (*lost == QmdLost::Mappings) {
        ++state.qmd_lost_mappings[static_cast<size_t>(kind)];
    }
}

/// A signal the QMD view could not value at once waits: the first by `why` (`kind`: what sent the
/// last write that may write anywhere or change the mappings), or behind one (`why` empty).
inline void NoteQmdWait(std::optional<QmdLost> why, UnplacedKind kind = UnplacedKind::Other) {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    if (!why) {
        ++state.qmd_behind;
        return;
    }
    ++state.qmd_waits[static_cast<size_t>(*why)];
    if (*why == QmdLost::Unplaced) {
        ++state.qmd_lost_unplaced[static_cast<size_t>(kind)];
    } else if (*why == QmdLost::Mappings) {
        ++state.qmd_lost_mappings[static_cast<size_t>(kind)];
    }
}

/// A signal waits for a write into a QMD by `writer`, still queued.
inline void NoteQmdWaitWriter(const std::string& writer) {
    if (g_enabled) {
        ++Get().qmd_wait_writers[writer];
    }
}

/// The QMD view valued a waiting signal late; `caught_up`: none waits now.
inline void NoteQmdLate(bool caught_up) {
    if (g_enabled) {
        State& state = Get();
        ++state.qmd_late;
        state.qmd_caught_up += caught_up ? 1 : 0;
    }
}

/// The QMD view lost its table valuing a waiting signal late (`why`), dropping `dropped` waiting
/// signals.
inline void NoteQmdLateLost(QmdLost why, size_t dropped) {
    if (g_enabled) {
        State& state = Get();
        ++state.qmd_late_lost[static_cast<size_t>(why)];
        state.qmd_late_dropped += dropped;
    }
}

/// A takeover dropped `dropped` waiting signals: the back had run them.
inline void NoteQmdTakeOverDropped(size_t dropped) {
    if (g_enabled) {
        Get().qmd_takeover_dropped += dropped;
    }
}

/// The QMD view read a descriptor from an inline QMD still queued.
inline void NoteQmdInlineRead() {
    if (g_enabled) {
        ++Get().qmd_inline_reads;
    }
}

/// The QMD view noted a release so.
inline void NoteQmdNote(QmdNote note) {
    if (g_enabled) {
        ++Get().qmd_notes[static_cast<size_t>(note)];
    }
}

/// The QMD view took the back's table over, at `where`, in `time`.
inline void NoteQmdSync(QmdTakeOver where, std::chrono::steady_clock::duration time) {
    if (g_enabled) {
        State& state = Get();
        ++state.qmd_syncs[static_cast<size_t>(where)];
        state.qmd_sync_time += time;
    }
}

/// An acquire of `operation` (Puller GpuSemaphoreOperation) at `device_address`, decided so.
/// Suspended: whether its word is on a GPU-written page, and the buffer cache held it newer.
inline void NoteAcquire(u32 operation, Decision decision, u64 device_address, bool waited,
                        bool gpu_written, bool flush) {
    State& state = Get();
    ++state.acquires[OperationIndex(operation)][static_cast<size_t>(decision)];
    if (waited) {
        ++state.positioned_waits;
    }
    if (decision == Decision::Queued) {
        const auto it = state.noted.find(device_address & ~u64{3});
        if (it != state.noted.end() && ((it->second >> static_cast<u32>(Releaser::Qmd)) & 1)) {
            ++state.passed_by_qmd;
        }
    }
    if (decision != Decision::Suspended) {
        return;
    }
    if (gpu_written) {
        ++state.suspended_gpu_written;
    }
    if (flush) {
        ++state.suspended_flush;
    }
    const auto it = state.noted.find(device_address & ~u64{3});
    if (it == state.noted.end()) {
        ++state.suspended_never_noted;
        return;
    }
    for (u32 releaser = 0; releaser < static_cast<u32>(Releaser::Count); ++releaser) {
        if ((it->second >> releaser) & 1) {
            ++state.suspended_noted[releaser];
        }
    }
}

/// A suspended acquire resumed, after `waited`.
inline void NoteResume(Resume how, Source latch_source, bool after_flush,
                       std::chrono::steady_clock::duration waited) {
    State& state = Get();
    ++state.resumes[static_cast<size_t>(how)];
    if (how == Resume::Latch) {
        ++state.latches[static_cast<size_t>(latch_source)];
    }
    if (after_flush) {
        ++state.resumes_after_flush;
    }
    const u64 ns = static_cast<u64>(std::max<std::chrono::nanoseconds::rep>(
        0, std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count()));
    state.wait_ns += ns;
    state.wait_max_ns = std::max(state.wait_max_ns, ns);
    if (ns > 1'000'000) {
        ++state.waits_over_1ms;
    }
}

/// The front, between commands.
inline void MaybeReport() {
    if (!g_enabled) {
        return;
    }
    State& state = Get();
    const auto now = std::chrono::steady_clock::now();
    if (state.last_report == std::chrono::steady_clock::time_point{}) {
        state.last_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - state.last_report).count();
    if (seconds < 10.0) {
        return;
    }
    state.last_report = now;
    const auto by_operation = [&](size_t operation) {
        const auto& counts = state.acquires[operation];
        u64 total = 0;
        for (const u64 count : counts) {
            total += count;
        }
        return fmt::format("{} (passed by memory {}, by a release still queued {}, after a "
                           "positioned wait {}; suspended {}; drained {})",
                           total, counts[0], counts[1], counts[2], counts[3], counts[4]);
    };
    const auto& noted = state.suspended_noted;
    const auto& resumes = state.resumes;
    const auto& latches = state.latches;
    const u64 resumed = resumes[0] + resumes[1] + resumes[2] + resumes[3];
    LOG_INFO(HW_GPU,
             "Acquires in {:.1f} s (EDEN_NX2_ACQUIRE_STATS): equal {}, greater or equal {}, mask "
             "{}; positioned waits {}; passed by a queued QMD launch's release {} | suspended at "
             "words on GPU-written pages {} (held newer by the buffer cache {}), with releases "
             "sent by the puller {}, 3D reports {}, compute reports {}, the copy engine {}, QMD "
             "launches the front valued {}, none (other QMD releases, the CPU) {} | resumed {}: "
             "latched {} (payload releases {}, query results {}, QMD reductions {}, QMD maxima {}, "
             "other {}), by memory {}, by a queued release {}, by the drain check {}, after a "
             "flush {} | waited {:.1f} ms, at most {:.3f} ms, over 1 ms {}",
             seconds, by_operation(0), by_operation(1), by_operation(2), state.positioned_waits,
             state.passed_by_qmd, state.suspended_gpu_written, state.suspended_flush, noted[0],
             noted[1], noted[2], noted[3], noted[4], state.suspended_never_noted, resumed,
             resumes[0], latches[1], latches[2], latches[3], latches[4], latches[0], resumes[1],
             resumes[2], resumes[3], state.resumes_after_flush,
             static_cast<double>(state.wait_ns) / 1e6, static_cast<double>(state.wait_max_ns) / 1e6,
             state.waits_over_1ms);
    const auto& lost = state.qmd_lost;
    const auto& notes = state.qmd_notes;
    const auto& syncs = state.qmd_syncs;
    const auto kinds = [](const auto& counts) {
        std::string text;
        for (size_t kind = 0; kind < counts.size(); ++kind) {
            if (counts[kind] != 0) {
                text += fmt::format("{}{} {}", text.empty() ? " (" : ", ",
                                    unplaced_kind_names[kind], counts[kind]);
            }
        }
        return text.empty() ? text : text + ")";
    };
    // The writers a signal waited for most.
    std::vector<std::pair<u64, std::string>> writers;
    for (const auto& [writer, count] : state.qmd_wait_writers) {
        writers.emplace_back(count, writer);
    }
    std::sort(writers.rbegin(), writers.rend());
    std::string by_writer;
    for (size_t i = 0; i < writers.size() && i < 6; ++i) {
        by_writer +=
            fmt::format("{}{} {}", i == 0 ? " (by " : ", ", writers[i].second, writers[i].first);
    }
    if (!by_writer.empty()) {
        by_writer += ")";
    }
    const auto& waits = state.qmd_waits;
    const auto& late_lost = state.qmd_late_lost;
    const auto index = [](QmdLost why) { return static_cast<size_t>(why); };
    u64 waited_first = 0;
    u64 lost_late = 0;
    for (size_t why = 0; why < waits.size(); ++why) {
        waited_first += waits[why];
        lost_late += late_lost[why];
    }
    LOG_INFO(
        HW_GPU,
        "QMD view in {:.1f} s (EDEN_NX2_ACQUIRE_STATS): signals valued {}, valued late {} "
        "(caught up {} times) | waited first {} (a write into the QMD queued {}{}, work "
        "that may write anywhere queued {}{}, a change of the mappings queued {}{}), behind an "
        "earlier one {}; lost while waiting {} (a QMD written after it {}, the mappings "
        "changed after it {}, a release into a QMD read late {}, too many waiting {}; {} "
        "waiting dropped), dropped at takeovers {} | not valued: table not synced yet {}, "
        "QMD on a GPU-written page {}, a release of the signal into it {} | descriptors read "
        "from an inline QMD still queued {} | releases noted exact {}, at least {}, unknown "
        "{} | tables taken over {} in {:.3f} ms (at a front's start {}, between commands {}, "
        "at QMD methods {})",
        seconds, state.qmd_valued, state.qmd_late, state.qmd_caught_up, waited_first,
        waits[index(QmdLost::Pending)], by_writer, waits[index(QmdLost::Unplaced)],
        kinds(state.qmd_lost_unplaced), waits[index(QmdLost::Mappings)],
        kinds(state.qmd_lost_mappings), state.qmd_behind, lost_late,
        late_lost[index(QmdLost::WrittenAfter)], late_lost[index(QmdLost::MappedAfter)],
        late_lost[index(QmdLost::LateReleasedInto)], late_lost[index(QmdLost::TooMany)],
        state.qmd_late_dropped, state.qmd_takeover_dropped, lost[index(QmdLost::NotSynced)],
        lost[index(QmdLost::GpuWritten)], lost[index(QmdLost::ReleasedInto)],
        state.qmd_inline_reads, notes[0], notes[1], notes[2], syncs[0] + syncs[1] + syncs[2],
        std::chrono::duration<double, std::milli>(state.qmd_sync_time).count(), syncs[0], syncs[1],
        syncs[2]);
    auto noted_words = std::move(state.noted);
    state = State{};
    state.noted = std::move(noted_words);
    state.last_report = now;
}

} // namespace VideoCommon::Nx2AcquireStats
