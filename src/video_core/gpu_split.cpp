// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <thread>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstring>
#include <string>
#include <string_view>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_stats.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon {

namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};

/// TEST: an engine's name, by the class it was bound as (MethodEndingAt).
[[nodiscard]] std::string_view EngineName(std::optional<Tegra::Engines::EngineTypes> type) {
    using Tegra::Engines::EngineTypes;
    if (!type) {
        return "an unbound engine";
    }
    switch (*type) {
    case EngineTypes::Nv01Timer:
        return "Nv01Timer";
    case EngineTypes::KeplerCompute:
        return "KeplerCompute";
    case EngineTypes::Maxwell3D:
        return "Maxwell3D";
    case EngineTypes::Fermi2D:
        return "Fermi2D";
    case EngineTypes::MaxwellDMA:
        return "MaxwellDMA";
    case EngineTypes::KeplerMemory:
        return "KeplerMemory";
    }
    return "an engine";
}

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (!instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        stack += fmt::format(" {}+{:#x}", base_name ? base_name + 1 : name,
                             reinterpret_cast<uintptr_t>(frames[i]) -
                                 reinterpret_cast<uintptr_t>(module));
    }
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on thread {} while the back runs; stack:{}",
              where, thread, stack);
#else
    (void)where;
#endif
}

GpuSplit GpuSplit::instance;

void GpuSplit::MaybeStart(Core::System& system_) {
    static const char mode = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT");
        return value != nullptr ? value[0] : '2';
    }();
    const bool wanted = mode == '1' || mode == '2';
    if (!wanted || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    // The front reads command lists straight from guest memory; a read that flushes the caches
    // first (safe DMA, accurate GPU) would touch back-owned state. The settings can change
    // mid-game: hand the engines back to the front (and start again when they allow it).
    const bool safe_reads = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh()
                                                          : Settings::IsDMALevelSafe();
    const bool allowed = !safe_reads && !Settings::IsGPULevelHigh() &&
                         Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "GPU thread split: stopped, the accuracy settings need the caches on the front");
        return;
    }
    if (active || !allowed) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    profile_publishes = Nx2Profile::Enabled();                      // TEST
    log_methods = std::getenv("EDEN_NX2_ACQUIRE_STATS") != nullptr; // TEST
    ring = std::make_unique<u64[]>(RingWords);
    ++starts; // Before the back starts: it tags the entries of fenced writes with it (EntryTag)
    back_thread = std::jthread([this](std::stop_token stop_token) { BackLoop(stop_token); });
    active = true;
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode == '2' && (jit == nullptr || jit[0] != '0');
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}",
             front_macros ? ", macros on the first" : "");
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    back_thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
    back_thread.join();
    active = false;
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    last_writer_position = 0;
    last_sync_position = 0;
    last_entry_start = 0;
    back_entry_start = 0;
    semaphore_writes.clear();
    semaphore_writes_limit = 1024;
    last_any_write = 0;
    last_unplaced_write = 0;
    last_mapping_change = 0;
    last_unplaced_kind = UnplacedKind::Other;
    last_mapping_kind = UnplacedKind::Other;
    last_noted_write = 0;
    write_sets.fill({});
    coarse_sets.fill({});
    published_local = 0;
    publish_held = false;
    logged_method_count = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
    back_engine = nullptr;
    front_macros = false;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_idle_at.store(~u64{0}, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
    front_waiting.store(false, std::memory_order_relaxed);
    tick_requested.store(false, std::memory_order_relaxed);
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::NoteFirstUnpublished() {
    first_unpublished_ticks = Nx2Profile::Ticks();
}

void GpuSplit::NoteEarlyPublish() {
    Nx2Profile::Add(Nx2Profile::Counter::EarlyPublishes, 1);
}

void GpuSplit::NoteLatePublish() {
    const u64 idle_since = back_idle_since.load(std::memory_order_relaxed);
    if (idle_since == 0 || first_unpublished_ticks == 0) {
        return;
    }
    // The back could have run the words since it ran out of work or since the first was sent.
    const u64 since = std::max(idle_since, first_unpublished_ticks);
    const u64 now = Nx2Profile::Ticks();
    if (now > since) {
        Nx2Profile::Add(Nx2Profile::Counter::LatePublishTicks, now - since);
        Nx2Profile::Add(Nx2Profile::Counter::LatePublishes, 1);
    }
}

void GpuSplit::Publish() {
    if (published_local == write_position) {
        return;
    }
    if (profile_publishes) [[unlikely]] {
        NoteLatePublish(); // TEST
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void GpuSplit::Drain(DrainSite site) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            publish_every = PublishAfterWait;
            Nx2Profile::AddDrain(site, start);
            return;
        }
        Backoff(spins, write_position);
    }
}

bool GpuSplit::WaitForWriters() {
    return WaitFor(std::min(last_writer_position, write_position), DrainSite::ListWriters);
}

bool GpuSplit::WaitForSync() {
    return WaitFor(std::min(last_sync_position, write_position), DrainSite::ListSync);
}

bool GpuSplit::WaitForSyncedWriters(DrainSite site) {
    return WaitFor(std::min({last_writer_position, last_sync_position, write_position}), site);
}

void GpuSplit::NoteSemaphoreWrite(u64 device_address, u32 payload, bool known, bool at_least,
                                  Landing landing) {
    if (semaphore_writes.size() >= semaphore_writes_limit) {
        // The words of releases in memory hold their values.
        std::erase_if(semaphore_writes,
                      [this](const auto& entry) { return InMemory(entry.second); });
        semaphore_writes_limit = std::max<size_t>(1024, semaphore_writes.size() * 2);
    }
    // The release is in the last entry sent, or one before (a QMD signal valued late).
    SemaphoreWrite write{write_position, EntryTag(last_entry_start), payload,
                         known,          at_least,                   landing == Landing::Fenced};
    const u64 word = device_address & ~u64{3};
    if (const auto it = semaphore_writes.find(word);
        !write.fenced && it != semaphore_writes.end() && it->second.fenced &&
        fenced_landed.load(std::memory_order_acquire) <= it->second.entry) {
        // A fenced write sent before may land after this one: until it did, the word holds
        // either.
        write.entry = it->second.entry;
        write.known = false;
        write.at_least = false;
        write.fenced = true;
    }
    semaphore_writes.insert_or_assign(word, write);
}

std::optional<GpuSplit::QueuedSemaphoreWrite> GpuSplit::QueuedSemaphoreWriteAt(
    u64 device_address) const {
    const auto it = semaphore_writes.find(device_address & ~u64{3});
    if (it == semaphore_writes.end() || InMemory(it->second)) {
        return std::nullopt;
    }
    return QueuedSemaphoreWrite{it->second.payload, it->second.known, it->second.at_least,
                                it->second.fenced, it->second.position};
}

bool GpuSplit::WritersPending() const {
    return consumed.load(std::memory_order_acquire) <
           std::min(last_writer_position, write_position);
}

namespace {

/// Calls `func(device_address, size)` for each mapped piece of [address, address + size) within
/// one 4 KiB GPU page; nothing is read or written where no device memory is mapped.
template <typename Func>
void ForEachDeviceRange(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                        Func&& func) {
    constexpr GPUVAddr page_size = 0x1000;
    const GPUVAddr end = address + size;
    for (GPUVAddr begin = address; begin < end;) {
        const GPUVAddr page_end = std::min((begin & ~(page_size - 1)) + page_size, end);
        if (const auto device = memory.GpuToCpuAddress(begin)) {
            func(*device, page_end - begin);
        }
        begin = page_end;
    }
}

} // Anonymous namespace

void GpuSplit::NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) {
    NoteWriteAt(memory, address, size, write_position);
}

void GpuSplit::NoteWriteAt(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                           u64 position) {
    if (size == 0) {
        return;
    }
    if (size > MaxCoarseNotedBytes || address + size < address) {
        NoteUnplacedWrite();
        TagUnplaced(UnplacedKind::Oversized); // TEST
        return;
    }
    if (size > MaxNotedBytes) {
        ForEachDeviceRange(memory, address, size, [this, position](DAddr device, u64 bytes) {
            NoteCoarseDeviceWriteAt(device, bytes, position);
        });
        return;
    }
    ForEachDeviceRange(memory, address, size, [this, position](DAddr device, u64 bytes) {
        NoteDeviceWriteAt(device, bytes, position);
    });
}

void GpuSplit::NoteSetWrite(WriteSet& set, u64 granule, u64 position) {
    const u32 tag = static_cast<u32>(granule);
    size_t oldest = 0;
    for (size_t way = 0; way < WriteWays; ++way) {
        if (set.granules[way] == tag) {
            set.positions[way] = std::max(set.positions[way], position);
            return;
        }
        if (set.positions[way] < set.positions[oldest]) {
            oldest = way;
        }
    }
    // The granule's earlier writes, if any, are among the evicted ones.
    const u64 granule_before = set.evicted;
    set.evicted = std::max(set.evicted, set.positions[oldest]);
    set.granules[oldest] = tag;
    set.positions[oldest] = std::max(position, granule_before);
}

u64 GpuSplit::SetTarget(const WriteSet& set, u64 granule) {
    const u32 tag = static_cast<u32>(granule);
    for (size_t way = 0; way < WriteWays; ++way) {
        if (set.granules[way] == tag) {
            return set.positions[way];
        }
    }
    return set.evicted;
}

void GpuSplit::NoteDeviceWriteAt(DAddr address, u64 size, u64 position) {
    const u64 last = (address + size - 1) >> WriteGranuleBits;
    for (u64 granule = address >> WriteGranuleBits; granule <= last; ++granule) {
        NoteSetWrite(write_sets[WriteSlot(granule)], granule, position);
    }
    last_noted_write = std::max(last_noted_write, position);
}

void GpuSplit::NoteCoarseDeviceWriteAt(DAddr address, u64 size, u64 position) {
    const u64 last = (address + size - 1) >> CoarseGranuleBits;
    for (u64 granule = address >> CoarseGranuleBits; granule <= last; ++granule) {
        NoteSetWrite(coarse_sets[CoarseSlot(granule)], granule, position);
    }
    last_noted_write = std::max(last_noted_write, position);
}

u64 GpuSplit::WritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) const {
    if (size > MaxNotedBytes || address + size < address) {
        return std::max(last_any_write, last_noted_write);
    }
    u64 target = last_any_write;
    ForEachDeviceRange(memory, address, size, [this, &target](DAddr device, u64 bytes) {
        target = std::max(target, DeviceWritesTarget(device, bytes));
    });
    return target;
}

u64 GpuSplit::MemoryWritesTarget(const Tegra::MemoryManager& memory, GPUVAddr address,
                                 u64 size) const {
    u64 target = last_unplaced_write;
    if (size > MaxNotedBytes || address + size < address) {
        target = std::max(target, last_noted_write);
    } else {
        ForEachDeviceRange(memory, address, size, [this, &target](DAddr device, u64 bytes) {
            target = std::max(target, DeviceWritesTarget(device, bytes));
        });
    }
    return std::min(target, write_position);
}

u64 GpuSplit::DeviceWritesTarget(DAddr address, u64 size) const {
    u64 target = 0;
    const u64 last = (address + size - 1) >> WriteGranuleBits;
    for (u64 granule = address >> WriteGranuleBits; granule <= last; ++granule) {
        target = std::max(target, SetTarget(write_sets[WriteSlot(granule)], granule));
    }
    const u64 last_coarse = (address + size - 1) >> CoarseGranuleBits;
    for (u64 granule = address >> CoarseGranuleBits; granule <= last_coarse; ++granule) {
        target = std::max(target, SetTarget(coarse_sets[CoarseSlot(granule)], granule));
    }
    return target;
}

bool GpuSplit::WaitForWrites(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) {
    return WaitFor(std::min(WritesTarget(memory, address, size), write_position),
                   DrainSite::MacroDmaWrites);
}

bool GpuSplit::WaitFor(u64 target, DrainSite site) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return false;
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            // The back may run out of what was sent before long.
            publish_every = PublishAfterWait;
            Nx2Profile::AddDrain(site, start);
            return true;
        }
        Backoff(spins, target);
    }
}

void GpuSplit::Backoff(u32 spins, u64 target) {
    if (spins < 16384) {
        Pause();
        return;
    }
    if (!back_running.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        return;
    }
    // A long wait sleeps until the back stores a new position (port of #161's front_wake): the
    // back checks front_waiting after each store, the front checks the position after setting it.
    front_waiting.store(true, std::memory_order_seq_cst);
    const u32 seen = front_wake.load(std::memory_order_seq_cst);
    if (consumed.load(std::memory_order_seq_cst) < target) {
        front_wake.wait(seen, std::memory_order_seq_cst);
    }
    front_waiting.store(false, std::memory_order_relaxed);
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    front_engine = engine;
    u64* const entry = Reserve(2);
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment,
                           Writes writes) {
    SelectEngine(engine);
    // Long runs (constant buffer and inline uploads) go in pieces, as when a method's data
    // continues in the next GPFIFO entry.
    while (amount > 0) {
        const u32 piece = std::min(amount, MaxMultiWords);
        const size_t head = segment != nullptr ? 3 : 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
                   (u64{method & 0xffff} << 8) | (u64{piece} << 32);
        entry[1] = methods_pending;
        if (segment != nullptr) {
            entry[2] = *segment;
        }
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size);
        words += piece;
        amount -= piece;
        methods_pending -= piece;
    }
    if (log_methods) [[unlikely]] {
        LogMethod(engine, method); // TEST
    }
    NoteWrites(writes);
}

void GpuSplit::LogMethod(const Tegra::Engines::EngineInterface* engine, u32 method) {
    if (!logged_methods) {
        logged_methods = std::make_unique<SentMethod[]>(LoggedMethods);
    }
    logged_methods[logged_method_count++ & (LoggedMethods - 1)] = {write_position,
                                                                   engine->bound_type, method};
}

std::string GpuSplit::MethodEndingAt(u64 end) const {
    if (!logged_methods || logged_method_count == 0) {
        return {};
    }
    // Ends grow with the count: a binary search over the ones kept.
    u64 low = logged_method_count > LoggedMethods ? logged_method_count - LoggedMethods : 0;
    u64 high = logged_method_count;
    while (low < high) {
        const u64 middle = low + (high - low) / 2;
        if (logged_methods[middle & (LoggedMethods - 1)].end < end) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low == logged_method_count) {
        return {};
    }
    const SentMethod& sent = logged_methods[low & (LoggedMethods - 1)];
    if (sent.end != end) {
        return {};
    }
    // Not typeid: without RTTI (MSVC's /GR-) it reads type data the engines don't have.
    return fmt::format("{} {:#x}", EngineName(sent.engine), sent.method);
}

void GpuSplit::DeferFunction(std::function<void()>&& func, Writes writes) {
    auto* const heap = new std::function<void()>(std::move(func));
    Defer(
        [heap] {
            (*heap)();
            delete heap;
        },
        writes);
}

size_t GpuSplit::Execute(const u64* entry, size_t words) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        return 2;
    case Kind::Method: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        if (engine->fast_maxwell3d) {
            // Nx2Perf: as below without the profile zones (off with fast_maxwell3d) and the
            // virtual call.
            using Tegra::Engines::Maxwell3D;
            auto* const maxwell3d = static_cast<Maxwell3D*>(engine);
            constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
            if (method - cb_data < 16 && (head & HasSegment) == 0) {
                // Single constant buffer words in a row (macros write them one by one) go up in
                // one write. They are executable: each entry would only consume the sink first.
                std::array<u32, 64> methods;
                std::array<u32, 64> arguments;
                methods[0] = method;
                arguments[0] = argument;
                u32 count = 1;
                while (count < methods.size() && count < words) {
                    const u64 next = entry[count];
                    const u32 next_method = static_cast<u32>((next >> 8) & 0xffff);
                    if (static_cast<Kind>(next & 0xf) != Kind::Method || (next & HasSegment) != 0 ||
                        next_method - cb_data >= 16) {
                        break;
                    }
                    methods[count] = next_method;
                    arguments[count] = static_cast<u32>(next >> 32);
                    ++count;
                }
                maxwell3d->ConsumeSink(*system);
                if (count == 1) {
                    maxwell3d->CallMethod(*system, method, argument, true);
                } else {
                    maxwell3d->CallConstBufferWords(*system, methods.data(), arguments.data(),
                                                    count);
                }
                return count;
            }
            if ((head & Direct) != 0) {
                maxwell3d->ConsumeSink(*system);
                maxwell3d->CallMethod(*system, method, argument, true);
            } else if (!maxwell3d->execution_mask[method]) {
                maxwell3d->method_sink.emplace_back(method, argument);
            } else {
                maxwell3d->ConsumeSink(*system);
                if ((head & HasSegment) != 0) {
                    maxwell3d->current_dma_segment = entry[1];
                }
                maxwell3d->CallMethod(*system, method, argument, (head & IsLast) != 0);
            }
            return (head & HasSegment) != 0 ? 2 : 1;
        }
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        if ((head & Direct) != 0) {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            engine->CallMethod(*system, method, argument, true);
        } else if (!engine->execution_mask[method]) {
            engine->method_sink.emplace_back(method, argument);
        } else {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                engine->current_dma_segment = entry[1];
            }
            engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::Call: {
        if ((head & FastDrawNeutral) == 0) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::Call);
        }
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 2);
        return 2 + payload;
    }
    }
    return 1;
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuBack};
    Common::SetCurrentThreadName("GPUBack");
    on_back_thread = true;
    back_running.store(true, std::memory_order_release);
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    if (Settings::values.pin_gpu_threads.GetValue()) {
        Common::PinCurrentThread(Common::PinnedRole::GpuBack);
    }
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    while (!stop_token.stop_requested()) {
        Nx2Profile::Report("GPU back thread");
        MaybeTick();
        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {
            // The front publishes what it holds at its next commit (a few words) from now on.
            back_idle_at.store(read_position, std::memory_order_relaxed);
            Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait};
            const GpuWaitStats::IdleScope idle{GpuWaitStats::g_back_idle_ns};
            // TEST (profile_publishes): out of work from now until this block ends.
            struct IdleMark {
                std::atomic<u64>& since;
                ~IdleMark() {
                    since.store(0, std::memory_order_relaxed);
                }
            } const idle_mark{back_idle_since};
            if (profile_publishes) {
                back_idle_since.store(Nx2Profile::Ticks(), std::memory_order_relaxed);
            }
            // The front usually sends more within microseconds: spin briefly before sleeping.
            for (u32 spins = 0; spins < 8192 && available == read_position &&
                                !hook_requested.load(std::memory_order_relaxed) &&
                                !tick_requested.load(std::memory_order_relaxed);
                 ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position && idle_hook != nullptr) {
                hook_requested.store(false, std::memory_order_seq_cst);
                idle_hook(idle_argument);
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                // A request made after the hook ran (RequestIdleHook) runs it before sleeping, and
                // a requested tick runs at the top of the loop.
                if (available == read_position && !stop_token.stop_requested() &&
                    !hook_requested.load(std::memory_order_seq_cst) &&
                    !tick_requested.load(std::memory_order_seq_cst)) {
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        u64 last_store = read_position;
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            back_entry_start = read_position;
            const size_t size =
                Execute(&ring[index], std::min<u64>(available - read_position, RingWords - index));
            read_position += size != 0 ? size : RingWords - index;
            if (hook_requested.load(std::memory_order_relaxed) &&
                idle_hook != nullptr) [[unlikely]] {
                hook_requested.store(false, std::memory_order_seq_cst);
                idle_hook(idle_argument);
            }
            // A guest thread waits for a flush: run it now, not after the draws and compute work
            // still queued here, or after everything the front holds (the guest synchronized with
            // the GPU work it reads before it read, so that work ran on this thread already).
            MaybeTick();
            // Hand ring space back now and then while working through a long batch, and at once
            // when the front waits for this position (WaitForWriters).
            if ((read_position - last_store >= 4096 ||
                 read_position >= wait_target.load(std::memory_order_relaxed)) &&
                read_position != available) {
                consumed.store(read_position, std::memory_order_seq_cst);
                last_store = read_position;
                NotifyFront();
            }
        }
        consumed.store(read_position, std::memory_order_seq_cst);
        NotifyFront();
    }
    back_running.store(false, std::memory_order_seq_cst);
    front_wake.fetch_add(1, std::memory_order_seq_cst);
    front_wake.notify_all();
}

} // namespace VideoCommon
