// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2.h"

// EDEN_NX2_LIST_STATS: what the front's reads of command lists on GPU-written pages
// (DmaPusher::ReadGpuWrittenCommands) wait for. For each list its size; for each of its pages how
// BufferCache::Nx2ReadListCopy served it, the GPU work that last wrote the page (a dispatch's
// shader, a copy engine copy's source, or draws) and whether the copy held other bytes than guest
// memory; for each copy the front waited for: how long, how long before the back had recorded it,
// how much the back still had to run (and of that, not yet published) and whether other channels
// had lists queued. Every 10 s the front logs the counts.
//
// EDEN_NX2_LIST_VERIFY: each list read so is read again as the front read lists before it tracked
// the words GPU work wrote (after every dispatch and copy sent so far, from the pages' whole
// copies, or drained and flushed) and, when the two differ, drained and flushed as well, which
// tells the stale read. Differences are counted and the first ones logged; the front parses the
// words read last.
namespace Tegra::Nx2ListStats {

// Plain flags: checked at every GPU-written list and buffer write mark.
inline bool g_enabled = std::getenv("EDEN_NX2_LIST_STATS") != nullptr;
inline bool g_verify = std::getenv("EDEN_NX2_LIST_VERIFY") != nullptr;

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitches() {
    g_enabled = std::getenv("EDEN_NX2_LIST_STATS") != nullptr;
    g_verify = std::getenv("EDEN_NX2_LIST_VERIFY") != nullptr;
}

[[nodiscard]] inline bool Any() {
    return g_enabled || g_verify;
}

[[nodiscard]] inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline constexpr u64 PageBytes = u64{1} << Nx2WrittenPageBits;

/// How BufferCache::Nx2ReadListCopy served a list's bytes on one page.
enum class Served : u32 {
    PageSynced, ///< From guest memory: the front flushed the page, no GPU work wrote it since
    WordsClean, ///< From guest memory: no GPU work wrote the list's words since the flush
    CopyReady,  ///< From the page's copy, which the GPU had finished
    CopyWaited, ///< From the page's copy, once the GPU finished it
    Unknown,    ///< Nowhere: the front drains and flushes
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Served::Count)> served_names{
    "page synced", "words clean", "copy ready", "copy waited", "unknown",
};

/// The GPU work that last wrote a command list page (BufferCache::Nx2NoteListWrite).
enum class Writer : u32 {
    None,     ///< No write noted since the stats began
    Dispatch, ///< A compute dispatch: the id is its shader's hash
    Copy,     ///< A copy engine copy: the id is its source GPU address
    Draw,     ///< Draws and the other GPU writes
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Writer::Count)> writer_names{
    "none",
    "dispatch",
    "copy",
    "draw",
};

struct WriterTag {
    Writer writer{Writer::Draw};
    u64 id{};
};

/// The back: the dispatch or copy the GPU writes noted now come from.
inline thread_local WriterTag t_writer{};

/// The back, around a dispatch or a copy engine copy: tags the GPU writes noted meanwhile.
class WriterScope {
public:
    explicit WriterScope(Writer writer, u64 id) : previous{t_writer} {
        t_writer = WriterTag{writer, id};
    }
    ~WriterScope() {
        t_writer = previous;
    }
    WriterScope(const WriterScope&) = delete;
    WriterScope& operator=(const WriterScope&) = delete;

private:
    WriterTag previous;
};

/// What Nx2ReadListCopy told about its last read where its result does not (NoteServed).
struct ServedDetail {
    bool valid{};
    Served served{Served::Unknown};
    u64 wait_start_ns{}; ///< CopyWaited: when the front started to wait
    u64 waited_ns{};
};
inline thread_local ServedDetail t_served{};

/// The front, in Nx2ReadListCopy: how it served the read.
inline void NoteServed(Served served, u64 wait_start_ns = 0, u64 waited_ns = 0) {
    if (Any()) {
        t_served = ServedDetail{true, served, wait_start_ns, waited_ns};
    }
}

/// The front, while EDEN_NX2_LIST_VERIFY reads a list as before: Nx2ReadListCopy serves guest
/// memory only for pages flushed and written by no GPU work since, else the page's whole copy.
inline thread_local bool t_reference{};

class ReferenceScope {
public:
    ReferenceScope() {
        t_reference = true;
    }
    ~ReferenceScope() {
        t_reference = false;
    }
    ReferenceScope(const ReferenceScope&) = delete;
    ReferenceScope& operator=(const ReferenceScope&) = delete;
};

inline constexpr size_t ChannelSlots = 16;
/// Command lists submitted to the GPU thread (ThreadManager::SubmitList) that it has not taken
/// yet, by channel.
inline std::array<std::atomic<s32>, ChannelSlots> g_queued_lists{};

[[nodiscard]] inline size_t ChannelSlot(s32 channel) {
    return static_cast<size_t>(static_cast<u32>(channel)) % ChannelSlots;
}

/// Any thread: a list of `channel` was queued for the GPU thread (`delta` 1) or taken (-1).
inline void NoteListQueued(s32 channel, s32 delta) {
    if (g_enabled) {
        g_queued_lists[ChannelSlot(channel)].fetch_add(delta, std::memory_order_relaxed);
    }
}

/// Whether lists of channels other than `channel` wait for the GPU thread.
[[nodiscard]] inline bool OtherChannelsQueued(s32 channel) {
    const size_t own = ChannelSlot(channel);
    for (size_t slot = 0; slot < ChannelSlots; ++slot) {
        if (slot != own && g_queued_lists[slot].load(std::memory_order_relaxed) > 0) {
            return true;
        }
    }
    return false;
}

struct Waits {
    u64 reads{};
    u64 waits{};
    u64 wait_ns{};

    void Add(bool waited, u64 ns) {
        ++reads;
        if (waited) {
            ++waits;
            wait_ns += ns;
        }
    }
};

struct PageInfo {
    WriterTag writer{Writer::None, 0};
    u32 write_begin{}; ///< The last GPU write's bytes on the page
    u32 write_end{};
    u64 write_size{};        ///< The last GPU write's whole size
    u64 copy_ns{};           ///< When the back recorded the page's last copy
    Waits waits;             ///< This period's reads
    u32 list_begin{~u32{0}}; ///< The bytes of the page this period's lists read
    u32 list_end{};
};

struct WriterStats {
    WriterTag writer;
    Waits waits;  ///< This period's reads of the pages it wrote last
    u64 writes{}; ///< This period's GPU writes to command list pages
    u64 bytes{};  ///< Their bytes on those pages
};

/// How a list differed from its read as before (EDEN_NX2_LIST_VERIFY).
enum class Difference : u32 {
    OldStale,  ///< The drained read matches the new read: the read as before was stale
    NewStale,  ///< The drained read matches the read as before: the new read was stale
    AllDiffer, ///< All three differ
    Drained,   ///< The read as before drained and flushed: the new read was stale
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Difference::Count)> difference_names{
    "the read as before was stale",
    "the new read was stale",
    "all three differ",
    "the new read differs from the drained one",
};

/// One page of the list the front reads, for EDEN_NX2_LIST_VERIFY's log.
struct ListPart {
    u64 page{};
    u32 offset{};
    u32 size{};
    Served served{Served::Unknown};
};

inline constexpr std::array<u64, 5> list_size_limits{16, 64, 256, 1024, 4096};
inline constexpr std::array<u64, 4> wait_limits_us{100, 250, 500, 1000};

/// The counts of one logging period.
struct Period {
    u64 lists{};
    u64 list_words{};
    std::array<u64, list_size_limits.size() + 1> list_sizes{};
    std::array<u64, static_cast<size_t>(Served::Count)> served{};
    std::array<Waits, static_cast<size_t>(Writer::Count)> by_writer{};
    u64 after_sync{}; ///< Reads that first waited for the work before the last sync
    u64 wait_ns{};
    u64 wait_max_ns{};
    std::array<u64, wait_limits_us.size() + 1> wait_sizes{};
    u64 copies_compared{};
    u64 copies_equal{};
    u64 waits_backlog{}; ///< Copy waits the back had work for
    u64 backlog_words{};
    u64 waits_unpublished{}; ///< Copy waits with work the back did not see yet
    u64 unpublished_words{};
    u64 waits_other_channels{};
    u64 waits_recorded{}; ///< Copy waits whose copy's record time is known
    u64 since_recorded_ns{};

    u64 verified{};
    u64 verified_drained{}; ///< Read as before by draining and flushing
    u64 differ{};
    u64 differ_words{};
    std::array<u64, static_cast<size_t>(Difference::Count)> differences{};
};

struct State {
    std::mutex mutex;
    std::unordered_map<u64, PageInfo> pages;
    std::unordered_map<u64, WriterStats> writers;
    std::vector<ListPart> parts; ///< The pages of the list the front reads
    Period period;
    u64 verified_total{};
    u64 differ_total{};
    u32 differ_logged{};
    std::chrono::steady_clock::time_point last_log{std::chrono::steady_clock::now()};
};

inline State& Get() {
    static State state;
    return state;
}

[[nodiscard]] inline u64 WriterKey(const WriterTag& tag) {
    return (static_cast<u64>(tag.writer) << 60) ^ tag.id;
}

[[nodiscard]] inline std::string WriterText(const WriterTag& tag) {
    switch (tag.writer) {
    case Writer::Dispatch:
        return fmt::format("dispatch {:#018x}", tag.id);
    case Writer::Copy:
        return fmt::format("copy from {:#x}", tag.id);
    default:
        return writer_names[static_cast<size_t>(tag.writer)];
    }
}

/// The index of the first of `limits` `value` does not exceed, else limits.size().
[[nodiscard]] inline size_t Bucket(u64 value, std::span<const u64> limits) {
    size_t bucket = 0;
    while (bucket < limits.size() && value > limits[bucket]) {
        ++bucket;
    }
    return bucket;
}

[[nodiscard]] inline double Ms(u64 ns) {
    return static_cast<double>(ns) / 1e6;
}

[[nodiscard]] inline double Average(u64 total, u64 count, double unit) {
    return count == 0 ? 0.0 : static_cast<double>(total) / static_cast<double>(count) / unit;
}

/// The back: GPU work wrote [device_addr, device_addr + size) over the command list page at
/// `page_addr` (BufferCache::Nx2NoteListWrite).
inline void NoteWrite(u64 page_addr, u64 device_addr, u64 size) {
    const u64 begin = (std::max)(device_addr, page_addr) - page_addr;
    const u64 end = (std::min)(device_addr + size, page_addr + PageBytes) - page_addr;
    const WriterTag tag = t_writer;
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    PageInfo& page = state.pages[page_addr];
    page.writer = tag;
    page.write_begin = static_cast<u32>(begin);
    page.write_end = static_cast<u32>(end);
    page.write_size = size;
    WriterStats& stats = state.writers[WriterKey(tag)];
    stats.writer = tag;
    ++stats.writes;
    stats.bytes += end - begin;
}

/// The back recorded the copy of the command list page at `page_addr` (Nx2CopyListPages).
inline void NoteCopy(u64 page_addr) {
    const u64 now = NowNs();
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    state.pages[page_addr].copy_ns = now;
}

inline void LogStats(const State& state, double seconds) {
    const Period& counts = state.period;
    const auto writer_reads = [&counts](Writer kind) {
        const Waits& entry = counts.by_writer[static_cast<size_t>(kind)];
        return fmt::format("{} {}/{}/{:.1f}", writer_names[static_cast<size_t>(kind)], entry.reads,
                           entry.waits, Ms(entry.wait_ns));
    };
    const u64 copy_waits = counts.served[static_cast<size_t>(Served::CopyWaited)];
    LOG_INFO(
        HW_GPU,
        "list stats {:.1f}s: {} lists of {} words (up to 16 words {}, 64 {}, 256 {}, 1024 {}, "
        "4096 {}, more {}) | page reads: page synced {}, words clean {}, copy ready {}, copy "
        "waited {}, unknown {}; after a sync wait {} | copy waits {:.1f} ms, average {:.0f} "
        "us, max {:.0f} us (up to 100 us {}, 250 {}, 500 {}, 1000 {}, more {}) | by the "
        "page's last writer (reads/waits/ms): {}, {}, {}, {} | copies equal to guest memory "
        "{} of {} | at copy waits: the back had work {} (average {:.0f} words), not yet "
        "published {} (average {:.0f} words), other channels had lists queued {}, the copy "
        "was recorded {:.0f} us before on average ({} known)",
        seconds, counts.lists, counts.list_words, counts.list_sizes[0], counts.list_sizes[1],
        counts.list_sizes[2], counts.list_sizes[3], counts.list_sizes[4], counts.list_sizes[5],
        counts.served[0], counts.served[1], counts.served[2], counts.served[3], counts.served[4],
        counts.after_sync, Ms(counts.wait_ns), Average(counts.wait_ns, copy_waits, 1e3),
        static_cast<double>(counts.wait_max_ns) / 1e3, counts.wait_sizes[0], counts.wait_sizes[1],
        counts.wait_sizes[2], counts.wait_sizes[3], counts.wait_sizes[4],
        writer_reads(Writer::Dispatch), writer_reads(Writer::Copy), writer_reads(Writer::Draw),
        writer_reads(Writer::None), counts.copies_equal, counts.copies_compared,
        counts.waits_backlog, Average(counts.backlog_words, counts.waits_backlog, 1.0),
        counts.waits_unpublished, Average(counts.unpublished_words, counts.waits_unpublished, 1.0),
        counts.waits_other_channels, Average(counts.since_recorded_ns, counts.waits_recorded, 1e3),
        counts.waits_recorded);

    std::vector<const WriterStats*> writers;
    for (const auto& entry : state.writers) {
        writers.push_back(&entry.second);
    }
    std::ranges::sort(writers, [](const WriterStats* lhs, const WriterStats* rhs) {
        return std::tie(lhs->waits.wait_ns, lhs->waits.reads, lhs->writes) >
               std::tie(rhs->waits.wait_ns, rhs->waits.reads, rhs->writes);
    });
    std::string writers_text;
    for (size_t i = 0; i < writers.size() && i < 8; ++i) {
        const WriterStats& stats = *writers[i];
        writers_text += fmt::format(
            "{}{}: {} reads, {} waits in {:.1f} ms, {} writes of {} bytes on list pages",
            writers_text.empty() ? "" : "; ", WriterText(stats.writer), stats.waits.reads,
            stats.waits.waits, Ms(stats.waits.wait_ns), stats.writes, stats.bytes);
    }
    if (!writers_text.empty()) {
        LOG_INFO(HW_GPU, "list stats writers: {}", writers_text);
    }

    std::vector<std::pair<u64, const PageInfo*>> pages;
    for (const auto& entry : state.pages) {
        if (entry.second.waits.reads != 0) {
            pages.emplace_back(entry.first, &entry.second);
        }
    }
    std::ranges::sort(pages, [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.second->waits.wait_ns, lhs.second->waits.reads) >
               std::tie(rhs.second->waits.wait_ns, rhs.second->waits.reads);
    });
    std::string pages_text;
    for (size_t i = 0; i < pages.size() && i < 8; ++i) {
        const PageInfo& info = *pages[i].second;
        pages_text += fmt::format(
            "{}{:#x}: {} reads, {} waits in {:.1f} ms, lists at {:#x}-{:#x}, last written "
            "{:#x}-{:#x} (of {} bytes) by {}",
            pages_text.empty() ? "" : "; ", pages[i].first, info.waits.reads, info.waits.waits,
            Ms(info.waits.wait_ns), info.list_begin, info.list_end, info.write_begin,
            info.write_end, info.write_size, WriterText(info.writer));
    }
    if (!pages_text.empty()) {
        LOG_INFO(HW_GPU, "list stats pages: {}", pages_text);
    }
}

inline void LogVerify(const State& state, double seconds) {
    const Period& counts = state.period;
    LOG_INFO(HW_GPU,
             "list verify {:.1f}s: {} lists read again ({} drained and flushed), {} differ in {} "
             "words: {} {}, {} {}, {} {}, {} {}; since the start {} of {} differ",
             seconds, counts.verified, counts.verified_drained, counts.differ, counts.differ_words,
             difference_names[0], counts.differences[0], difference_names[1], counts.differences[1],
             difference_names[2], counts.differences[2], difference_names[3], counts.differences[3],
             state.differ_total, state.verified_total);
}

/// The front: logs every 10 s.
inline void MaybeLog() {
    State& state = Get();
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock{state.mutex};
    const double seconds = std::chrono::duration<double>(now - state.last_log).count();
    if (seconds < 10.0) {
        return;
    }
    state.last_log = now;
    if (g_enabled) {
        LogStats(state, seconds);
    }
    if (g_verify) {
        LogVerify(state, seconds);
    }
    state.period = {};
    state.writers.clear();
    for (auto& entry : state.pages) {
        entry.second.waits = {};
        entry.second.list_begin = ~u32{0};
        entry.second.list_end = 0;
    }
}

/// The front starts reading a list of `words` on GPU-written pages.
inline void BeginList(u32 words) {
    MaybeLog();
    t_served = {};
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    state.parts.clear();
    ++state.period.lists;
    state.period.list_words += words;
    ++state.period.list_sizes[Bucket(words, list_size_limits)];
}

/// The front read `size` bytes at `offset` of the command list page at `page_addr`. `result`: as
/// Nx2ReadListCopy's result tells (PageSynced, CopyReady or Unknown; NoteServed tells more).
/// `after_sync`: it first waited for the work before the last sync (a page draws write).
/// `copy_equal`: for a read from the copy, whether guest memory held the same bytes. `backlog`
/// and `unpublished`: the ring words the back had not run before the read, and of them those it
/// did not see. `channel`: the list's.
inline void NoteRead(u64 page_addr, u32 offset, u32 size, Served result, bool after_sync,
                     std::optional<bool> copy_equal, u64 backlog, u64 unpublished, s32 channel) {
    const ServedDetail detail = t_served;
    t_served = {};
    const Served served = result == Served::Unknown || !detail.valid ? result : detail.served;
    const bool waited = served == Served::CopyWaited;
    const u64 wait_ns = waited ? detail.waited_ns : 0;
    const bool other_channels = waited && OtherChannelsQueued(channel);
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    state.parts.push_back(ListPart{page_addr, offset, size, served});
    if (!g_enabled) {
        return;
    }
    Period& counts = state.period;
    ++counts.served[static_cast<size_t>(served)];
    counts.after_sync += after_sync ? 1 : 0;
    PageInfo& page = state.pages[page_addr];
    page.waits.Add(waited, wait_ns);
    page.list_begin = (std::min)(page.list_begin, offset);
    page.list_end = (std::max)(page.list_end, offset + size);
    counts.by_writer[static_cast<size_t>(page.writer.writer)].Add(waited, wait_ns);
    if (page.writer.writer != Writer::None) {
        WriterStats& stats = state.writers[WriterKey(page.writer)];
        stats.writer = page.writer;
        stats.waits.Add(waited, wait_ns);
    }
    if (copy_equal) {
        ++counts.copies_compared;
        counts.copies_equal += *copy_equal ? 1 : 0;
    }
    if (!waited) {
        return;
    }
    counts.wait_ns += wait_ns;
    counts.wait_max_ns = (std::max)(counts.wait_max_ns, wait_ns);
    ++counts.wait_sizes[Bucket(wait_ns / 1000, wait_limits_us)];
    if (backlog != 0) {
        ++counts.waits_backlog;
        counts.backlog_words += backlog;
    }
    if (unpublished != 0) {
        ++counts.waits_unpublished;
        counts.unpublished_words += unpublished;
    }
    counts.waits_other_channels += other_channels ? 1 : 0;
    if (page.copy_ns != 0 && page.copy_ns <= detail.wait_start_ns) {
        ++counts.waits_recorded;
        counts.since_recorded_ns += detail.wait_start_ns - page.copy_ns;
    }
}

/// EDEN_NX2_LIST_VERIFY: the front read the list at `address` as `parsed`, then as before as
/// `reference` (`reference_drained`: by draining and flushing) and, when those differ, drained
/// and flushed as `drained` (else empty).
inline void NoteVerify(u64 address, std::span<const u32> parsed, std::span<const u32> reference,
                       std::span<const u32> drained, bool reference_drained) {
    u64 differing = 0;
    size_t first = parsed.size();
    for (size_t i = 0; i < parsed.size() && i < reference.size(); ++i) {
        if (parsed[i] != reference[i]) {
            ++differing;
            first = (std::min)(first, i);
        }
    }
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    Period& counts = state.period;
    ++counts.verified;
    ++state.verified_total;
    counts.verified_drained += reference_drained ? 1 : 0;
    if (differing == 0) {
        return;
    }
    Difference kind = Difference::Drained;
    if (!reference_drained) {
        if (std::ranges::equal(drained, parsed)) {
            kind = Difference::OldStale;
        } else if (std::ranges::equal(drained, reference)) {
            kind = Difference::NewStale;
        } else {
            kind = Difference::AllDiffer;
        }
    }
    ++counts.differ;
    ++state.differ_total;
    counts.differ_words += differing;
    ++counts.differences[static_cast<size_t>(kind)];
    if (state.differ_logged >= 32) {
        return;
    }
    ++state.differ_logged;
    std::string parts_text;
    for (const ListPart& part : state.parts) {
        const auto it = state.pages.find(part.page);
        const PageInfo info = it != state.pages.end() ? it->second : PageInfo{};
        parts_text += fmt::format("{}{:#x}+{:#x} ({} bytes) {}, last written {:#x}-{:#x} by {}",
                                  parts_text.empty() ? "" : "; ", part.page, part.offset, part.size,
                                  served_names[static_cast<size_t>(part.served)], info.write_begin,
                                  info.write_end, WriterText(info.writer));
    }
    LOG_WARNING(HW_GPU,
                "list verify: list at {:#x} ({} words) differs from its read as before in {} "
                "words, the first word {}: {:#010x}, as before {:#010x}{}: {}; its pages: {}",
                address, parsed.size(), differing, first, parsed[first], reference[first],
                drained.size() > first ? fmt::format(", drained {:#010x}", drained[first])
                                       : std::string{},
                difference_names[static_cast<size_t>(kind)], parts_text);
}

} // namespace Tegra::Nx2ListStats
