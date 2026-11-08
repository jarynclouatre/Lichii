// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/cache_types.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/rasterizer_interface.h"

// TEST: checks of the command entries DmaPusher::Step parses without the GPU-written path, from
// their submit-time copy or from guest memory.
//
// EDEN_NX2_DRAIN_CHECK: the entries that began between methods and so did not wait for the
// compute work sent before the last sync while some was queued. At each one's place the back
// looks for pages that work has made GPU-written by then, brings the GPU's bytes of those to guest
// memory and compares the words the front parsed: differing ones must be 0.
//
// EDEN_NX2_ENTRY_CHECK: every such entry compared at its place on the back with guest memory:
// words written after the front took them, by the CPU or by GPU work that writes guest memory
// past the caches (inline uploads, semaphore releases, reports), which no page bit tells about.
//
// One line every 10 s from the GPU thread.
namespace Tegra::Nx2EntryCheck {

// A plain flag: checked at every command entry.
inline bool g_entries = std::getenv("EDEN_NX2_ENTRY_CHECK") != nullptr;

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_entries = std::getenv("EDEN_NX2_ENTRY_CHECK") != nullptr;
}

[[nodiscard]] inline bool Any() {
    return g_entries || VideoCommon::Nx2DrainCheck::g_enabled;
}

/// What the front parsed: the first words and a hash of all.
struct Parsed {
    std::array<u32, 6> head{};
    u32 words{};
    u64 hash{};
};

[[nodiscard]] inline Parsed Summarize(std::span<const u32> words) {
    Parsed parsed;
    parsed.words = static_cast<u32>(words.size());
    parsed.hash = HashPushbufferWords(words);
    for (size_t i = 0; i < parsed.head.size() && i < words.size(); ++i) {
        parsed.head[i] = words[i];
    }
    return parsed;
}

/// The words of guest memory at [address, address + words * 4).
[[nodiscard]] inline Parsed SummarizeMemory(MemoryManager& memory, GPUVAddr address, u32 words) {
    std::vector<u32> data(words);
    if (words != 0) {
        memory.ReadBlockUnsafe(address, data.data(), data.size() * sizeof(u32));
    }
    return Summarize(data);
}

enum Path : u32 { Copy, Memory, PathCount };

struct State {
    // Front.
    std::atomic<u64> skipped{};         ///< Entries that did not wait for queued writers
    std::atomic<u64> kept_for_copies{}; ///< Command entries that waited for queued copy work
    std::array<std::atomic<u64>, PathCount> compared{};
    // Back.
    std::atomic<u64> skipped_checked{};
    std::atomic<u64> skipped_written{}; ///< On a page GPU work had made GPU-written at its place
    std::atomic<u64> skipped_differ{};  ///< Of those, words differing after a flush
    std::array<std::atomic<u64>, PathCount> differ{};
    std::atomic<u32> logged{};
    std::chrono::steady_clock::time_point last_log{std::chrono::steady_clock::now()};
};

inline State& Get() {
    static State state;
    return state;
}

[[nodiscard]] inline std::string Hex(std::span<const u32> words, u32 count) {
    std::string text;
    for (u32 i = 0; i < count && i < words.size(); ++i) {
        text += fmt::format("{}{:08x}", text.empty() ? "" : " ", words[i]);
    }
    return text.empty() ? std::string{"-"} : text;
}

inline void LogDiffer(const char* what, GPUVAddr address, const Parsed& parsed,
                      const Parsed& memory) {
    State& state = Get();
    if (state.logged.fetch_add(1, std::memory_order_relaxed) >= 16) {
        return;
    }
    LOG_WARNING(HW_GPU,
                "Command entry check: {} at {:#x} ({} words): parsed starts {}, guest memory "
                "{} at the back's place",
                what, address, parsed.words, Hex(parsed.head, parsed.words),
                Hex(memory.head, memory.words));
}

/// Back, at the place of an entry that did not wait for the writers before the last sync: were
/// its words written by that work?
inline void CheckSkipped(MemoryManager* memory, VideoCore::RasterizerInterface* rasterizer,
                         GPUVAddr address, Parsed parsed) {
    State& state = Get();
    state.skipped_checked.fetch_add(1, std::memory_order_relaxed);
    const GPUVAddr end = address + u64{parsed.words} * sizeof(u32);
    bool written = false;
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < end; page += 0x1000) {
        const auto device = memory->GpuToCpuAddress(page);
        written = written || (device && Nx2IsGpuWritten(*device));
    }
    if (!written) {
        return;
    }
    state.skipped_written.fetch_add(1, std::memory_order_relaxed);
    if (rasterizer != nullptr) {
        // The GPU's bytes of the entry's pages, in guest memory for the comparison.
        for (GPUVAddr piece = address; piece < end;) {
            const GPUVAddr piece_end = (std::min)(end, (piece & ~GPUVAddr{0xfff}) + 0x1000);
            if (const auto device = memory->GpuToCpuAddress(piece)) {
                rasterizer->FlushRegion(*device, piece_end - piece,
                                        VideoCommon::CacheType::BufferCache);
            }
            piece = piece_end;
        }
    }
    const Parsed now = SummarizeMemory(*memory, address, parsed.words);
    if (now.hash != parsed.hash) {
        state.skipped_differ.fetch_add(1, std::memory_order_relaxed);
        LogDiffer("an entry that did not wait for the writers differs from GPU-written memory",
                  address, parsed, now);
    }
}

/// Back, at an entry's place (EDEN_NX2_ENTRY_CHECK): does guest memory still hold what the front
/// parsed?
inline void CheckCompared(MemoryManager* memory, GPUVAddr address, Parsed parsed, Path path) {
    const Parsed now = SummarizeMemory(*memory, address, parsed.words);
    if (now.hash == parsed.hash) {
        return;
    }
    Get().differ[path].fetch_add(1, std::memory_order_relaxed);
    LogDiffer(path == Copy ? "an entry run from its submit-time copy differs from guest memory"
                           : "an entry run from guest memory differs from guest memory later",
              address, parsed, now);
}

inline void MaybeLog() {
    State& state = Get();
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - state.last_log).count();
    if (seconds < 10.0) {
        return;
    }
    state.last_log = now;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 skipped = take(state.skipped);
    const u64 kept = take(state.kept_for_copies);
    const u64 checked = take(state.skipped_checked);
    const u64 written = take(state.skipped_written);
    const u64 differ = take(state.skipped_differ);
    const u64 copies = take(state.compared[Copy]);
    const u64 copies_differ = take(state.differ[Copy]);
    const u64 reads = take(state.compared[Memory]);
    const u64 reads_differ = take(state.differ[Memory]);
    LOG_INFO(HW_GPU,
             "Command entry checks in {:.1f} s: entries that did not wait for queued writers {} "
             "({} checked at their place, on pages GPU work had made GPU-written {}, their words "
             "differing after a flush {}), command entries that waited for queued copy work {} | "
             "entries compared with guest memory at their place: run from their submit-time copy "
             "{} ({} differ), from guest memory {} ({} differ)",
             seconds, skipped, checked, written, differ, kept, copies, copies_differ, reads,
             reads_differ);
}

/// Front, before it parses the entry `header` of `pusher` from its submit-time copy (`copy`, else
/// guest memory). `skipped_writers`: the entry did not wait for the writers before the last sync;
/// `kept_for_copies`: a command entry that waited for them because copy work was queued.
inline void NoteEntry(DmaPusher& pusher, const CommandListHeader& header,
                      std::span<const CommandHeader> copy, bool skipped_writers,
                      bool kept_for_copies) {
    MaybeLog();
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active()) {
        return;
    }
    const GPUVAddr address = header.addr.Value();
    const auto words = static_cast<u32>(header.size.Value());
    Parsed parsed;
    if (!copy.empty()) {
        std::vector<u32> data(copy.size());
        for (size_t i = 0; i < copy.size(); ++i) {
            data[i] = copy[i].argument;
        }
        parsed = Summarize(data);
    } else {
        parsed = SummarizeMemory(pusher.memory_manager, address, words);
    }
    State& state = Get();
    if (VideoCommon::Nx2DrainCheck::g_enabled) {
        if (kept_for_copies && split.WritersPending()) {
            state.kept_for_copies.fetch_add(1, std::memory_order_relaxed);
        }
        if (skipped_writers && split.WritersPending()) {
            state.skipped.fetch_add(1, std::memory_order_relaxed);
            split.Defer([memory = &pusher.memory_manager, rasterizer = pusher.rasterizer, address,
                         parsed] { CheckSkipped(memory, rasterizer, address, parsed); },
                        VideoCommon::GpuSplit::Writes::Caches);
        }
    }
    if (g_entries) {
        const Path path = copy.empty() ? Memory : Copy;
        state.compared[path].fetch_add(1, std::memory_order_relaxed);
        split.Defer([memory = &pusher.memory_manager, address, parsed,
                     path] { CheckCompared(memory, address, parsed, path); },
                    VideoCommon::GpuSplit::Writes::Noted);
    }
}

} // namespace Tegra::Nx2EntryCheck
