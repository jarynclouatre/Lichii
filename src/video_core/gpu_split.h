// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {
class EngineInterface;
enum class EngineTypes : u32;
}

namespace VideoCommon {

/// TEST (EDEN_NX2_ACQUIRE_STATS): what sent the last write the front could not place
/// (GpuSplit::TagUnplaced), for why the QMD view could not value a signal.
enum class UnplacedKind : u8 {
    Other,      ///< A deferred call nobody tagged
    RunCommand, ///< A function the CPU queued in order (unmaps, events)
    Engine,     ///< An engine method the front sends as it is (blits)
    Oversized,  ///< A placed write longer than the coarse slots take
    QmdLaunch,  ///< A QMD launch the view could not value
    InlineQmd,  ///< An inline QMD behind a change of the mappings
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(UnplacedKind::Count)>
    unplaced_kind_names{"other",        "CPU functions", "engines", "oversized writes",
                        "QMD launches", "inline QMDs"};

/// Why the split's front waited for its back (EDEN_NX2_GPU_PROFILE counts each).
enum class DrainSite : u32 {
    Other,
    FetchDrain,     ///< EDEN_NX2_FETCH_DRAIN: before every entry
    Acquire,        ///< A semaphore acquire reads the value the back may release
    SemaphoreFlush, ///< The acquired semaphore is in the buffer cache
    FenceRelease,   ///< The puller's acquires release the pending fences
    ListWriters,    ///< GPU-written command lists: the dispatches and copies sent before them
    ListSync,       ///< Command lists on draw-written pages: the work before the last host sync
    ListFlush,      ///< Command lists on GPU-written pages: drain and flush the pages
    MacroDmaRead,   ///< The front's macros read memory (MME DMA) the caches may still write
    MacroDmaWrites, ///< Those reads: the queued work that can write the words (WaitForWrites)
    MacroDmaFlush,  ///< Those words are in the buffer cache
    GpSyncFence,    ///< A GP entry's sync bit (sync_memory_operations)
    Map,            ///< GPU memory maps before later command reads
    FrontStart,     ///< The front takes an engine's macros over
    FrontVerify,    ///< EDEN_NX2_FRONT_VERIFY
    BackRegister,   ///< A register a macro the back ran may have written (Maxwell3D)
    BackCode,       ///< The macro code changes under the macros the back has yet to run
    ListCopies,     ///< Lists on GPU-written pages: the dispatches and copies before the last sync
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DrainSite::Count)> drain_site_names{
    "other",           "fetch drain",   "acquire",    "semaphore flush", "fence release",
    "list writers",    "list sync",     "list flush", "macro dma read",  "macro dma writes",
    "macro dma flush", "gp sync fence", "map",        "front start",     "front verify",
    "back register",   "back code",     "list copy writers",
};

// Splits the GPU thread in two (NX2_GPU_THREAD_SPLIT.md). The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// How often the split started: state the front copied during an earlier start is stale.
    [[nodiscard]] u32 Starts() const {
        return starts;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// Whether the calling thread is the back.
    [[nodiscard]] static bool OnBackThread() {
        return on_back_thread;
    }
    /// Back thread: the entry it runs, as NoteFencedWrite takes it; elsewhere nothing.
    [[nodiscard]] std::optional<u64> BackEntry() const {
        if (!on_back_thread) {
            return std::nullopt;
        }
        return EntryTag(back_entry_start);
    }
    /// Any thread: the operation of a report the back sent in `entry` (BackEntry) ran with its
    /// fence, so the report is in memory, and those of the entries before it: fences run their
    /// operations in order (QueryCacheBase::CounterReport).
    void NoteFencedWrite(u64 entry) {
        u64 landed = fenced_landed.load(std::memory_order_relaxed);
        while (landed <= entry &&
               !fenced_landed.compare_exchange_weak(landed, entry + 1, std::memory_order_release,
                                                    std::memory_order_relaxed)) {
        }
    }
    /// Called on the back before it sleeps waiting for work (the rasterizer submits deferred work).
    void SetIdleHook(void (*hook)(void*), void* argument) {
        idle_hook = hook;
        idle_argument = argument;
    }
    /// Any thread: whether the back thread runs (false before it starts and once it stopped).
    [[nodiscard]] bool BackRunning() const {
        return back_running.load(std::memory_order_acquire);
    }
    /// Any thread: has the back call the idle hook soon, between entries or at once if it sleeps.
    void RequestIdleHook() {
        hook_requested.store(true, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
    /// Run on the back when a thread asks for it (RequestTick): the GPU's sync operations.
    void SetTickHook(void (*hook)(void*), void* argument) {
        tick_hook = hook;
        tick_argument = argument;
    }
    /// Any thread, after queueing a sync operation it waits for: has the back run the tick hook
    /// after the entry it is running, or at once if it waits for work. Otherwise the operation runs
    /// after the back's next draw, or after the front sent everything it had queued before it.
    void RequestTick() {
        if (!back_running.load(std::memory_order_acquire)) {
            return;
        }
        // The back checks tick_requested after setting back_waiting: one of the two sees the other.
        tick_requested.store(true, std::memory_order_seq_cst);
        if (back_waiting.load(std::memory_order_seq_cst)) {
            wake.fetch_add(1, std::memory_order_seq_cst);
            wake.notify_one();
        }
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// What the sender of an entry tells about the guest memory the back writes running it, for
    /// the front's reads of guest memory (WaitForWrites, MemoryWritesTarget).
    enum class Writes : u8 {
        /// Unknown: a read anywhere waits for the entry, and it may change the GPU's mappings
        /// (unmaps; GPU maps drain the split)
        Anywhere,
        /// Guest memory anywhere, the mappings not: engine work whose writes nobody placed
        Unplaced,
        /// The GPU caches anywhere (shader stores, copies into buffers), guest memory only through
        /// them, where GPU work wrote (Nx2MayBeGpuWritten): a read of the caches waits for it, a
        /// read of other guest memory not (MemoryWritesTarget)
        Caches,
        Noted, ///< Nothing, or what the sender notes right after sending it (NoteWrite)
    };

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment, Writes writes = Writes::Anywhere) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words);
        if (log_methods) [[unlikely]] {
            LogMethod(engine, method); // TEST
        }
        NoteWrites(writes);
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment,
                     Writes writes = Writes::Anywhere);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument,
                    Writes writes = Writes::Anywhere) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1);
        if (log_methods) [[unlikely]] {
            LogMethod(engine, method); // TEST
        }
        NoteWrites(writes);
    }

    /// Defer flag: the call doesn't end fast draw reuse (nx2_fast_draw.h).
    static constexpr u64 FastDrawNeutral = 1 << 7;

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    /// `flags`: FastDrawNeutral when `func` can't change what a draw resolves to.
    template <typename Func>
    void Defer(const Func& func, Writes writes = Writes::Anywhere, u64 flags = 0) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | flags | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload);
        NoteWrites(writes);
    }
    /// Defer, calling `func` with a copy of `words` (a std::span<const u32>).
    template <typename Func>
    void DeferWords(const Func& func, std::span<const u32> words,
                    Writes writes = Writes::Anywhere) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t function_words = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        const size_t payload = function_words + 1 + (words.size() + 1) / 2;
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallWordsThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        entry[2 + function_words] = words.size();
        if (!words.empty()) {
            entry[1 + payload] = 0; // Padding after an odd count
            std::memcpy(entry + 3 + function_words, words.data(), words.size_bytes());
        }
        Commit(2 + payload);
        NoteWrites(writes);
    }
    void DeferFunction(std::function<void()>&& func, Writes writes = Writes::Anywhere);
    /// Runs `func` on the back and waits for it: for work on back-owned state, such as a buffer
    /// cache download that records into the Vulkan scheduler. A drained back still submits from
    /// its idle hook, so the front never touches that state itself.
    template <typename Func>
    void Run(const Func& func, DrainSite site = DrainSite::Other) {
        if (on_back_thread || !active) {
            func();
            return;
        }
        Defer(func);
        Drain(site);
    }

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();
    /// The front waited for the back, which may run out of work soon: publish the next words
    /// after a few, as after a drain, not after PublishEvery.
    void PublishSoon() {
        publish_every = PublishAfterWait;
    }
    /// Keeps the entries the front sends unpublished while it notes what they write from guest
    /// memory as the back will find it running them (the QMD view: descriptors, and the words
    /// maxima and reductions combine with). An idle back gets a published entry at once and may
    /// run it before the front reads, which then reads what the entry wrote. While `hold`, Commit
    /// publishes nothing; Publish (drains, waits, a full ring) still does.
    class PublishHold {
    public:
        PublishHold(GpuSplit& split_, bool hold) : split{split_}, was_held{split_.publish_held} {
            if (hold) {
                split.publish_held = true;
            }
        }
        ~PublishHold() {
            if (split.publish_held != was_held) {
                split.publish_held = was_held;
                split.PublishIfDue();
            }
        }
        PublishHold(const PublishHold&) = delete;
        PublishHold& operator=(const PublishHold&) = delete;

    private:
        GpuSplit& split;
        bool was_held;
    };
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);

    /// Compute or copy work was just sent: what can write GPU-written command lists.
    void NoteWriter() {
        last_writer_position = write_position;
    }
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();
    /// A host-level wait was just sent (a syncpoint acquire, a host WFI, a GP entry with its sync
    /// bit; semaphore acquires note it when a release still queued meets them): the guest's GPU
    /// work before it is complete before the commands after it are fetched.
    void NoteSync() {
        last_sync_position = write_position;
    }
    /// The position after everything sent so far, for Ran and WaitForPosition.
    [[nodiscard]] u64 Position() const {
        return write_position;
    }
    /// Whether the back executed everything sent before `position`. It may tell so a little
    /// late: the back stores its position every few thousand words and when it runs out of work.
    [[nodiscard]] bool Ran(u64 position) const {
        return consumed.load(std::memory_order_acquire) >= position;
    }
    /// The position after the last compute or copy work sent (NoteWriter).
    [[nodiscard]] u64 LastWriterPosition() const {
        return (std::min)(last_writer_position, write_position);
    }
    /// Waits until the back executed everything sent before `position`, not what came after.
    /// Returns whether any of it was still queued.
    bool WaitForPosition(u64 position, DrainSite site) {
        return WaitFor((std::min)(position, write_position), site);
    }
    /// Waits until the back executed the work sent before the last NoteSync. Returns whether any
    /// of it was still queued.
    bool WaitForSync();
    /// Waits until the back executed the compute and copy work sent before the last NoteSync:
    /// the writers a command list fetched now is ordered after. Returns whether any of it was
    /// still queued. `site`: whose wait it is, for EDEN_NX2_GPU_PROFILE.
    bool WaitForSyncedWriters(DrainSite site = DrainSite::ListWriters);
    /// When a release the back runs reaches memory.
    enum class Landing : u8 {
        AtRun,  ///< As the back runs it: payload reports without a fence, reductions, maxima
        Fenced, ///< Once the fence signaled after it ran (QueryCacheBase::CounterReport):
                ///< fence-flagged reports, counter reports
    };
    /// A semaphore release was just sent: once it landed (`landing`), the word at
    /// `device_address` holds `payload`, or a value the front can't tell (`known` false: a
    /// reduction, a counter report), or `payload` at least (`at_least`: a maximum with a value the
    /// front can't tell).
    void NoteSemaphoreWrite(u64 device_address, u32 payload, bool known, bool at_least = false,
                            Landing landing = Landing::AtRun);
    struct QueuedSemaphoreWrite {
        u32 payload;
        bool known;
        bool at_least; ///< Known only as a lower bound (unsigned)
        /// It lands once a fence ran, at no position the front can wait for; or an earlier write
        /// that does may still land after it.
        bool fenced;
        u64 position; ///< Once the back ran this (Ran), the word holds the write, unless `fenced`
    };
    /// The last write sent to the semaphore word at `device_address`, if it may not be in memory
    /// yet.
    [[nodiscard]] std::optional<QueuedSemaphoreWrite> QueuedSemaphoreWriteAt(
        u64 device_address) const;
    /// Whether compute or copy work sent so far is still queued (EDEN_NX2_DRAIN_CHECK).
    [[nodiscard]] bool WritersPending() const;
    /// TEST (EDEN_NX2_ACQUIRE_STATS): the engine and method of the method entry that ends at
    /// `end`, among the last LoggedMethods sent, as "KeplerCompute 0x6d"; empty if none.
    [[nodiscard]] std::string MethodEndingAt(u64 end) const;
    /// Front thread: the ring words sent that the back had not run when it last stored its
    /// position (every 4096 words of a batch and at its end) (EDEN_NX2_LIST_STATS).
    [[nodiscard]] u64 Backlog() const {
        return write_position - consumed.load(std::memory_order_acquire);
    }
    /// Front thread: the ring words sent that the back does not see yet (EDEN_NX2_LIST_STATS).
    [[nodiscard]] u64 Unpublished() const {
        return write_position - published_local;
    }

    /// The entry just sent (Writes::Noted) writes [address, address + size) of `memory`, as
    /// mapped now. Up to MaxNotedBytes by 256-byte granule, up to MaxCoarseNotedBytes by 64 KiB
    /// granule, longer writes count as unplaced.
    void NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);
    /// NoteWrite for the entry sent at `position` (the one before it ends there), placed late: the
    /// front's QMD view values a launch once the back ran what it waited for.
    void NoteWriteAt(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size, u64 position);
    /// The entry just sent may write guest memory anywhere and change the GPU's mappings
    /// (Writes::Anywhere).
    void NoteAnyWrite() {
        NoteUnplacedWrite();
        last_mapping_change = write_position;
        last_mapping_kind = UnplacedKind::Other; // TEST
    }
    /// The entry just sent may write guest memory anywhere (Writes::Unplaced).
    void NoteUnplacedWrite() {
        last_any_write = write_position;
        last_unplaced_write = write_position;
        last_unplaced_kind = UnplacedKind::Other; // TEST
    }
    /// TEST: what sent the entry just sent, if it may write anywhere or change the mappings.
    void TagUnplaced(UnplacedKind kind) {
        if (last_unplaced_write == write_position) {
            last_unplaced_kind = kind;
        }
        if (last_mapping_change == write_position) {
            last_mapping_kind = kind;
        }
    }
    /// TEST: what sent the last entry that may write anywhere (LastUnplacedWrite), and the last
    /// that may change the mappings (LastMappingChange).
    [[nodiscard]] UnplacedKind LastUnplacedKind() const {
        return last_unplaced_kind;
    }
    [[nodiscard]] UnplacedKind LastMappingKind() const {
        return last_mapping_kind;
    }
    /// The entry just sent may write the GPU caches anywhere (Writes::Caches).
    void NoteCacheWrite() {
        last_any_write = write_position;
    }
    void NoteWrites(Writes writes) {
        if (writes == Writes::Anywhere) {
            NoteAnyWrite();
        } else if (writes == Writes::Unplaced) {
            NoteUnplacedWrite();
        } else if (writes == Writes::Caches) {
            NoteCacheWrite();
        }
    }
    /// The position after the last entry sent that may write guest memory where the front does
    /// not know (NoteAnyWrite, NoteUnplacedWrite), outside what the caches hold.
    [[nodiscard]] u64 LastUnplacedWrite() const {
        return (std::min)(last_unplaced_write, write_position);
    }
    /// The position after the last entry sent that may change the GPU's mappings (NoteAnyWrite):
    /// once the back ran it, what is mapped now is what the back finds mapped.
    [[nodiscard]] u64 LastMappingChange() const {
        return (std::min)(last_mapping_change, write_position);
    }
    /// Once the back ran everything before the position returned, no entry sent so far writes
    /// guest memory in [address, address + size) of `memory` but through the GPU caches
    /// (Writes::Caches): what the back reads there without them, as the QMD scheduler does, is in
    /// memory then. Placed as WaitForWrites places writes.
    [[nodiscard]] u64 MemoryWritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address,
                                         u64 size) const;
    /// Waits until the back executed every entry sent so far that can write guest memory in
    /// [address, address + size) of `memory`, not the others. Returns whether one was queued.
    /// Writes are placed by the device memory the GPU addresses map to when noted and read: GPU
    /// maps drain the split first (GPU::MapInOrder), the unmaps it is sent count as anywhere.
    bool WaitForWrites(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

private:
    friend struct GpuSplitTestAccess;

    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
    };
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;
    static constexpr size_t PublishAfterWait = 8;

    /// NoteWrite places writes by 256-byte granule of device memory, up to MaxNotedBytes long,
    /// in sets of WriteWays slots hashed by granule. Writes up to MaxCoarseNotedBytes (copies,
    /// uploads) go by 64 KiB granule into coarse sets, longer ones count as unplaced. A slot holds
    /// a granule and the position of its last write; a write to a granule its set lacks takes the
    /// slot written longest ago, and the set keeps the latest position of the writes it so
    /// dropped (`evicted`). A read waits for its granule's own last write while the set holds
    /// it, else for the evicted ones: it may wait for a write elsewhere, never miss one, and the
    /// writes to other granules hashed there (the constant buffer words of queued draws) hold up
    /// a read of a QMD only once they pushed it out of its set.
    static constexpr u32 WriteGranuleBits = 8;
    static constexpr u64 MaxNotedBytes = 0x4000;
    static constexpr u32 WriteSlotBits = 12; ///< Sets
    static constexpr u32 CoarseGranuleBits = 16;
    static constexpr u64 MaxCoarseNotedBytes = u64{16} << 20;
    static constexpr u32 CoarseSlotBits = 10;
    static constexpr size_t WriteWays = 4;

    /// One cache line.
    struct WriteSet {
        std::array<u64, WriteWays> positions;
        /// The granules' low 32 bits: granules that agree in them share a slot.
        std::array<u32, WriteWays> granules;
        u64 evicted;
        u64 unused;
    };
    static_assert(sizeof(WriteSet) == 64);
    /// The set a granule is hashed to.
    static size_t WriteSlot(u64 granule) {
        return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ULL) >> (64 - WriteSlotBits));
    }
    static size_t CoarseSlot(u64 granule) {
        return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ULL) >> (64 - CoarseSlotBits));
    }
    /// A write to `granule` at `position` (the newest, or an earlier one placed late).
    static void NoteSetWrite(WriteSet& set, u64 granule, u64 position);
    /// The latest position of the writes to `granule` in its set, at most.
    [[nodiscard]] static u64 SetTarget(const WriteSet& set, u64 granule);
    void NoteDeviceWrite(DAddr address, u64 size) {
        NoteDeviceWriteAt(address, size, write_position);
    }
    void NoteDeviceWriteAt(DAddr address, u64 size, u64 position);
    void NoteCoarseDeviceWrite(DAddr address, u64 size) {
        NoteCoarseDeviceWriteAt(address, size, write_position);
    }
    void NoteCoarseDeviceWriteAt(DAddr address, u64 size, u64 position);
    /// The position WaitForWrites waits for.
    [[nodiscard]] u64 WritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address,
                                   u64 size) const;
    [[nodiscard]] u64 DeviceWritesTarget(DAddr address, u64 size) const;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }
    template <typename Func>
    static void CallWordsThunk(const u64* payload) {
        constexpr size_t function_words = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        (*reinterpret_cast<const Func*>(payload))(
            std::span<const u32>{reinterpret_cast<const u32*>(payload + function_words + 1),
                                 static_cast<size_t>(payload[function_words])});
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words) {
        if (profile_publishes && write_position == published_local) [[unlikely]] {
            NoteFirstUnpublished(); // TEST
        }
        last_entry_start = write_position;
        write_position += words;
        if (PublishDue() && !publish_held) {
            PublishNow();
        }
    }
    /// Whether Commit publishes now.
    [[nodiscard]] bool PublishDue() const {
        const u64 unpublished = write_position - published_local;
        // The back waits for work at the last position published (it stored it in back_idle_at
        // before it spun): it gets these words after a few, not after publish_every.
        return unpublished >= publish_every ||
               (unpublished >= PublishAfterWait &&
                back_idle_at.load(std::memory_order_relaxed) == published_local);
    }
    void PublishNow() {
        if (profile_publishes && write_position - published_local < publish_every) [[unlikely]] {
            NoteEarlyPublish(); // TEST
        }
        Publish();
    }
    /// What Commit would have published during a PublishHold.
    void PublishIfDue() {
        if (PublishDue()) {
            PublishNow();
        }
    }
    /// TEST (EDEN_NX2_GPU_PROFILE): how long the back waited for work while the front held words
    /// it could have run (Publish).
    /// Nx2Profile::Enabled() as the split started (MaybeStart, after VideoCore::CreateGPU read
    /// the switch).
    bool profile_publishes{};
    void NoteFirstUnpublished();
    void NoteLatePublish();
    /// TEST: Commit publishes before publish_every, because the back waits for the words.
    void NoteEarlyPublish();
    void WaitForSpace(size_t words);
    /// TEST (EDEN_NX2_ACQUIRE_STATS): the engine method entries sent, by the position after them.
    struct SentMethod {
        u64 end;
        std::optional<Tegra::Engines::EngineTypes> engine; ///< The class it was bound as
        u32 method;
    };
    static constexpr size_t LoggedMethods = size_t{1} << 16;
    /// EDEN_NX2_ACQUIRE_STATS as the split started (MaybeStart).
    bool log_methods{};
    void LogMethod(const Tegra::Engines::EngineInterface* engine, u32 method);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    bool WaitFor(u64 target, DrainSite site);

    /// One step of a front wait for the back to consume `target`.
    void Backoff(u32 spins, u64 target);
    /// After the back stored `consumed`: wakes a front sleeping in Backoff.
    void NotifyFront() {
        if (front_waiting.load(std::memory_order_seq_cst)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            front_wake.notify_one();
        }
    }

    void BackLoop(std::stop_token stop_token);
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its size.
    size_t Execute(const u64* entry, size_t words);
    /// The back, between entries: runs the tick hook if a thread asked for it.
    void MaybeTick() {
        if (tick_requested.load(std::memory_order_relaxed) &&
            tick_requested.exchange(false, std::memory_order_acq_rel) && tick_hook != nullptr)
            [[unlikely]] {
            tick_hook(tick_argument);
        }
    }

    Core::System* system{};
    bool active{};
    u32 starts{};
    bool front_macros{};
    std::unique_ptr<u64[]> ring;

    // Front side.
    u64 write_position{};
    u64 last_writer_position{};
    u64 last_sync_position{};
    u64 last_entry_start{}; ///< Where the last entry sent starts
    struct SemaphoreWrite {
        u64 position; ///< write_position once it was sent
        /// `fenced`: the entry (EntryTag) whose fenced write must have landed (NoteFencedWrite)
        u64 entry;
        u32 payload;
        bool known;
        bool at_least;
        bool fenced;
    };
    /// The front's NoteSemaphoreWrite, by word address.
    std::unordered_map<u64, SemaphoreWrite> semaphore_writes;
    size_t semaphore_writes_limit = 1024; ///< Forget what the back ran past this many
    /// After the last entry that may write guest memory or the GPU caches anywhere (NoteAnyWrite,
    /// NoteUnplacedWrite, NoteCacheWrite), the last one that may write guest memory itself
    /// anywhere (NoteAnyWrite, NoteUnplacedWrite), the last one that may change the GPU's
    /// mappings (NoteAnyWrite), the last one NoteWrite placed, and the last placed write in each
    /// slot's granules, fine and coarse.
    u64 last_any_write{};
    u64 last_unplaced_write{};
    u64 last_mapping_change{};
    UnplacedKind last_unplaced_kind{}; ///< TEST: see TagUnplaced
    UnplacedKind last_mapping_kind{};  ///< TEST
    u64 last_noted_write{};
    alignas(64) std::array<WriteSet, size_t{1} << WriteSlotBits> write_sets{};
    alignas(64) std::array<WriteSet, size_t{1} << CoarseSlotBits> coarse_sets{};
    u64 published_local{};
    /// PublishHold: Commit publishes nothing.
    bool publish_held{};
    u64 consumed_cache{};
    u64 first_unpublished_ticks{}; ///< TEST: when the first word not published yet was sent
    std::unique_ptr<SentMethod[]> logged_methods; ///< TEST: see LogMethod
    u64 logged_method_count{};
    /// Words between publishes: small right after the front waited for the back (Drain,
    /// PublishSoon), while the back may have nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};
    u64 back_entry_start{}; ///< Where the entry the back runs starts (BackEntry)

    /// An entry's position as fenced writes name it: with this start of the split in the high
    /// bits, so a fence of an earlier start that runs late lands nothing of this one (positions
    /// stay far below 2^48).
    [[nodiscard]] u64 EntryTag(u64 position) const {
        return (u64{starts} << 48) | position;
    }
    /// Whether the word holds `write` (as far as no other write changed it since): the back ran
    /// it, and a fenced one landed.
    [[nodiscard]] bool InMemory(const SemaphoreWrite& write) const {
        return write.position <= consumed.load(std::memory_order_acquire) &&
               (!write.fenced || fenced_landed.load(std::memory_order_acquire) > write.entry);
    }

    static GpuSplit instance;
    static inline thread_local bool on_back_thread{};
    void (*idle_hook)(void*){};
    void* idle_argument{};
    void (*tick_hook)(void*){};
    void* tick_argument{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> tick_requested{};
    std::atomic<bool> back_running{};

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// One past the last entry (EntryTag) whose fenced writes landed (NoteFencedWrite): the
    /// fencing thread writes it, the front reads it.
    alignas(128) std::atomic<u64> fenced_landed{};
    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    /// The front sleeps in Backoff (#161: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    /// The back's position when it last ran out of work, stored before it waits for more: while
    /// it equals published_local, the back waits for the words the front holds (Commit). The back
    /// writes it once per wait, the front reads it at every Commit.
    alignas(128) std::atomic<u64> back_idle_at{~u64{0}};
    /// TEST (profile_publishes): when the back last ran out of work, 0 while it runs.
    alignas(128) std::atomic<u64> back_idle_since{};
    std::jthread back_thread;
};

} // namespace VideoCommon
