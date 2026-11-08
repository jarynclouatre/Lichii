// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/nx2_profile.h"
#include "video_core/dma_pusher.h"
#include "video_core/command_fetch_order.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/engine_upload.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_entry_check.h"
#include "video_core/nx2_list_indirect.h"
#include "video_core/nx2_list_indirect_stats.h"
#include "video_core/nx2_list_stats.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra {

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 ComputeInline = 0x6D;

namespace {

/// The split's position after the last copy engine method that may write memory (the front's).
u64 g_last_copy_writer{};

/// Whether copy engine work sent so far may still be queued on the split's back. A new session
/// starts the split's positions over: a position past the current one is from an older session.
[[nodiscard]] bool CopyWritersPending(const VideoCommon::GpuSplit& split) {
    if (g_last_copy_writer > split.Position()) {
        g_last_copy_writer = 0;
    }
    return !split.Ran(g_last_copy_writer);
}

/// EDEN_NX2_LIST_STATS and _VERIFY: the front read `words` of a command list at `address`
/// (`device`) on a GPU-written page, as `where` tells; see Nx2ListStats::NoteRead.
void NoteListRead(MemoryManager& memory, s32 channel, GPUVAddr address, u64 device,
                  std::span<const u8> words, VideoCore::RasterizerInterface::GpuWrittenList where,
                  bool after_sync, u64 backlog, u64 unpublished) {
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    using Nx2ListStats::Served;
    std::optional<bool> copy_equal;
    if (Nx2ListStats::g_enabled && where == GpuWrittenList::Copied) {
        std::vector<u8> guest(words.size());
        memory.ReadBlockUnsafe(address, guest.data(), guest.size());
        copy_equal = std::ranges::equal(guest, words);
    }
    const Served result = where == GpuWrittenList::Copied   ? Served::CopyReady
                          : where == GpuWrittenList::Synced ? Served::PageSynced
                                                            : Served::Unknown;
    constexpr u64 page_mask = Nx2ListStats::PageBytes - 1;
    Nx2ListStats::NoteRead(device & ~page_mask, static_cast<u32>(device & page_mask),
                           static_cast<u32>(words.size()), result, after_sync, copy_equal, backlog,
                           unpublished, channel);
}

static_assert(DmaPusher::DmaRegisterCount == Engines::MaxwellDMA::RegisterCount);
static_assert(DmaPusher::InlineToMemoryRegisterCount == Engines::KeplerMemory::Regs::NUM_REGS);

/// What the back writes running `method` of an engine the front sends methods to as they are
/// (GpuSplit::Writes). A method the engine only keeps for later (its method sink) writes nothing
/// yet. Copy launches write the caches where they read GPU-written memory; what they copy to and
/// release, and what inline-to-memory uploads write, DmaPusher::NoteEngineWrites places.
VideoCommon::GpuSplit::Writes EngineWrites(const Engines::EngineInterface& engine,
                                           Engines::EngineTypes type, u32 method) {
    using Writes = VideoCommon::GpuSplit::Writes;
    if (!engine.execution_mask[method]) {
        return Writes::Noted;
    }
    switch (type) {
    case Engines::EngineTypes::MaxwellDMA:
        return Writes::Caches;
    case Engines::EngineTypes::KeplerMemory:
        return Writes::Noted;
    default:
        // Blits and the engines without a front: where they write, the front does not know.
        return Writes::Unplaced;
    }
}

} // Anonymous namespace

bool DmaPusher::IsMemoryWriter(const Engines::EngineInterface& engine, Engines::EngineTypes type,
                              u32 method) {
    if (type == Engines::EngineTypes::KeplerCompute) {
        return static_cast<const Engines::KeplerCompute&>(engine).MayWriteMemory(method);
    }
    if (type == Engines::EngineTypes::MaxwellDMA) {
        return method >= MacroRegistersStart || engine.execution_mask[method];
    }
    return false;
}

DmaPusher::DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : system{system_}
    , memory_manager{memory_manager_}
    , channel_state{channel_state_}
    , signal_sync{false}
    , synced{false}
{}

DmaPusher::~DmaPusher() = default;

bool DmaPusher::CanResumeAcquire() {
    if (pending_list_tick) {
        return rasterizer->IsGpuTickDone(*pending_list_tick);
    }
    return puller.CanResumeAcquire(*this);
}

void DmaPusher::DispatchCalls() {
    // The sub-index is kept: a channel suspended on a semaphore resumes inside its command list.
    dma_state.is_last_call = true;
    while (system.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        // Submits, and releases the fences done (downloads where the GPU wrote).
        split.Defer(
            [gpu = &system.GPU()] {
                gpu->FlushCommands();
                gpu->OnCommandListEnd();
            },
            VideoCommon::GpuSplit::Writes::Caches);
        split.Publish();
        return;
    }
    system.GPU().FlushCommands();
    system.GPU().OnCommandListEnd();
}

bool DmaPusher::Step() {
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }
    // Waiting for the GPU's copy of a command list this channel reads next (ReadGpuWrittenCommands).
    if (pending_list_tick) {
        if (!rasterizer->IsGpuTickDone(*pending_list_tick)) {
            return false;
        }
        pending_list_tick.reset();
        resumed_list_read = true;
    }
    // Only for the entry that suspended: an acquire still waiting, the safe-read path or a stopped
    // split must not leave it set for a later list (which would then skip its writer wait).
    SCOPE_EXIT {
        resumed_list_read = false;
    };
    const u64 suspensions_before = suspensions.load(std::memory_order_relaxed);
    if (!puller.ResumeAcquire(*this)) {
        return false;
    }
    // Lists queued while the channel waited were read from memory, not as submitted: the game
    // writes the commands a list runs before it releases the acquire, so the words at submit time
    // could be older. Reading them as late as the parse is too late in Hyrule Warriors: Age of
    // Imprisonment fights: a command ring the game had already refilled for a later frame, read
    // as this frame's commands (methods on unbound subchannels, garbage QMDs, then a channel
    // waiting forever on a release it never got). The acquire is released now, so the commands
    // are written and the memory not yet reused: copy them here. EDEN_NX2_RESNAPSHOT=0: off.
    // The current list's entries before this one ran already, and an entry the acquire suspended
    // inside reads its remaining words from memory anyway (CanUseSubmittedCommands).
    if (ResnapshotEnabled() && suspensions.load(std::memory_order_relaxed) != suspensions_before) {
        const u64 now = suspensions.load(std::memory_order_acquire);
        size_t first_entry = dma_pushbuffer_subindex + (command_resume_offset != 0 ? 1 : 0);
        for (CommandList& list : dma_pushbuffer) {
            if (!list.submitted_offsets.empty()) {
                RefreshSnapshot(list, now, first_entry);
            }
            first_entry = 0;
        }
    }

    CommandList& command_list = dma_pushbuffer.front();

    const size_t prefetch_size = command_list.prefetch_command_list.size();
    const size_t command_list_size = command_list.command_lists.size();

    if (prefetch_size == 0 && command_list_size == 0) {
        dma_pushbuffer.pop_front();
        dma_pushbuffer_subindex = 0;
        return true;
    }

    if (prefetch_size > 0) {
        const std::span<const CommandHeader> commands{command_list.prefetch_command_list};
        command_resume_offset +=
            ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        if (puller.HasPendingAcquire()) {
            return false;
        }
        command_resume_offset = 0;
        dma_pushbuffer.pop_front();
        return true;
    }

    auto& current_command = command_list.command_lists[dma_pushbuffer_subindex];
    const CommandListHeader& header = current_command;
    dma_state.dma_get = header.addr;

    if (signal_sync && !synced) {
        std::unique_lock lk(sync_mutex);
        sync_cv.wait(lk, [this]() { return synced.load(); });
        signal_sync = false;
        synced = false;
    }

    if (header.size > 0) {
        if (header.sync && g_nx2_ampere.load(std::memory_order_relaxed)) {
            VideoCommon::GpuSplit::Get().NoteSync();
        }
        // EDEN_NX2_FETCH_DRAIN=1: wait for the back before every entry, as before the split
        // overlapped the two halves again.
        static const bool fetch_drain = std::getenv("EDEN_NX2_FETCH_DRAIN") != nullptr;
        if (fetch_drain && g_nx2_ampere.load(std::memory_order_relaxed)) {
            VideoCommon::GpuSplit::Get().Drain(VideoCommon::DrainSite::FetchDrain);
        }
        bool gpu_written = IsGpuWrittenCommands(dma_state.dma_get,
                                                 size_t{header.size} * sizeof(u32));
        if (subchannels[dma_state.subchannel] && dma_state.method_count) {
            const auto engine = subchannel_type[dma_state.subchannel];
            const bool kepler_payload = engine == Engines::EngineTypes::KeplerCompute && dma_state.method == ComputeInline && dma_state.non_incrementing;
            const bool macro_payload = engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart;
            if (kepler_payload || macro_payload) {
                const size_t words = std::min<size_t>(dma_state.method_count, header.size);
                auto* const engine_state = subchannels[dma_state.subchannel];
                if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
                    // The caches behind IsMemoryDirty belong to the split's back thread.
                    split.Defer(
                        [engine_state, memory = &memory_manager,
                         address = GPUVAddr{dma_state.dma_get}, bytes = words * sizeof(u32)] {
                            engine_state->current_dirty = memory->IsMemoryDirty(address, bytes);
                        },
                        VideoCommon::GpuSplit::Writes::Noted,
                        VideoCommon::GpuSplit::FastDrawNeutral);
                } else {
                    engine_state->current_dirty = memory_manager.IsMemoryDirty(dma_state.dma_get, words * sizeof(u32));
                }
            }
        }
        if (command_resume_offset == 0) {
            // A method's data continues across GP entries, whatever their level: title 4 ends a
            // main entry after a macro's first word and supplies the remaining parameters in a
            // 5-word subroutine entry. EDEN_NX2_PARK_SPLIT_METHODS restores the old guess (an
            // entry elsewhere is a call that starts clean; resume where the split one ended).
            static const bool park_split_methods = std::getenv("EDEN_NX2_PARK_SPLIT_METHODS") != nullptr;
            if (g_nx2_ampere) {
                if (static u32 split_logs{}; dma_state.method_count != 0 && split_logs < 64) {
                    ++split_logs;
                    LOG_DEBUG(HW_GPU, "Split method {:#x} ({} left) on ch {}: next entry {:#x} size {} raw {:#x} (level bit {}, priv bit {}), expected {:#x}, parked {}",
                             dma_state.method, dma_state.method_count, channel_state.bind_id,
                             u64{header.addr}, u64{header.size}, header.raw, u64{header.is_push_buffer},
                             u64{header.allow_flush}, split_method_end, parked_methods.size());
                    std::array<u32, 8> next_words{};
                    std::array<u32, 4> prev_words{};
                    memory_manager.ReadBlockUnsafe(header.addr, next_words.data(),
                                             std::min<size_t>(header.size, 8) * sizeof(u32));
                    memory_manager.ReadBlockUnsafe(split_method_end - sizeof(prev_words), prev_words.data(),
                                             sizeof(prev_words));
                    LOG_DEBUG(HW_GPU, "  previous entry ends {:08x}; next entry starts {:08x}",
                              fmt::join(prev_words, " "), fmt::join(next_words, " "));
                }
                if (park_split_methods) {
                    ResolveSplitMethod(header.addr);
                }
            }
            if (dma_pushbuffer_subindex < command_list.submit_hashes.size()) {
                CheckSubmitHash(header, command_list.submit_hashes[dma_pushbuffer_subindex]);
            }
        }
        // A resumed entry is read again: the words after the acquire may be new by now.
        // A safe read flushes the caches, which belong to the split's back thread while it runs
        // (the split only starts with unsafe reads; the settings can change mid-game).
        const bool use_safe = !VideoCommon::GpuSplit::Get().Active() &&
                              (Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh() : Settings::IsDMALevelSafe());
        // Pages no GPU work has written yet may still get the list from compute or copy work
        // queued before it. The GPU fetches the list after that work only past a host-level wait
        // (NoteSync): later work is not ordered before the fetch, and the list does not wait for
        // it (nor for draws, which write memory too; draw-written lists wait for the last sync).
        // Lists that begin between methods don't wait for compute work either: they are command
        // buffers the CPU writes. The words GPU work writes for the front to fetch are the
        // parameters of a method begun in the entry before (an indirect dispatch's or draw's,
        // as in Ocarina of Time and title 1), which begin inside its data. An entry resumed after
        // an acquire resumes after a puller method, which no parameter list holds. The copy
        // engine can copy GPU-written words anywhere: while its work is queued, every list waits.
        // The lists that did not wait are checked on the back (EDEN_NX2_DRAIN_CHECK).
        const bool command_entry = dma_state.method_count == 0;
        const bool waits_for_writers =
            !command_entry || CopyWritersPending(VideoCommon::GpuSplit::Get());
        if (!gpu_written && VideoCommon::GpuSplit::Get().Active() && waits_for_writers &&
            VideoCommon::GpuSplit::Get().WaitForSyncedWriters()) {
            gpu_written =
                IsGpuWrittenCommands(dma_state.dma_get, size_t{header.size} * sizeof(u32));
        }
        if (VideoCommon::Nx2DrainCheck::g_enabled && !gpu_written && waits_for_writers)
            [[unlikely]] {
            CheckUnwaitedWriters(dma_state.dma_get, size_t{header.size} * sizeof(u32));
        }
        // Resume and GPU-written entries need live words, even after a flush cleared dirty state.
        const bool submitted = (command_list.snapshot_suspensions & 1) == 0 &&
                               command_list.snapshot_suspensions ==
                                   suspensions.load(std::memory_order_acquire) &&
                               VideoCommon::CanUseSubmittedCommands(
            command_resume_offset != 0,
            dma_pushbuffer_subindex < command_list.submitted_offsets.size() &&
                !std::ranges::binary_search(command_list.deferred_entries,
                                            static_cast<u32>(dma_pushbuffer_subindex)),
            gpu_written,
            use_safe && memory_manager.IsMemoryDirty(dma_state.dma_get,
                                                     header.size * sizeof(u32)));
        if (Nx2EntryCheck::Any() && !gpu_written && !use_safe) [[unlikely]] {
            Nx2EntryCheck::NoteEntry(
                *this, header,
                submitted ? std::span<const CommandHeader>{command_list.submitted_words.data() +
                                                               command_list.submitted_offsets
                                                                   [dma_pushbuffer_subindex],
                                                           header.size}
                          : std::span<const CommandHeader>{},
                !waits_for_writers, command_entry && waits_for_writers);
        }
        if (submitted) {
            const std::span<const CommandHeader> commands{
                command_list.submitted_words.data() +
                    command_list.submitted_offsets[dma_pushbuffer_subindex],
                header.size};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Submitted);
            command_resume_offset += ProcessCommands(commands, 0);
        } else if (use_safe) {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::SafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Safe);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else if (gpu_written && TakeGpuWrittenListWords(header)) {
            // The words stay on the host GPU: the call takes their placeholders (not recorded
            // for EDEN_NX2_PB_CHECK, which would find the entry's memory changed).
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else if (bool pending = false;
                   ReadGpuWrittenCommands(dma_state.dma_get, static_cast<u32>(header.size),
                                          &pending) ||
                   pending) {
            if (pending) {
                // Parsed again once the GPU copied the list; other channels run meanwhile.
                return false;
            }
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Copy);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else {
            FlushGpuWrittenCommands(dma_state.dma_get, size_t{header.size} * sizeof(u32));
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::UnsafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Slow);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        }
        if (puller.HasPendingAcquire()) {
            return false;
        }
        command_resume_offset = 0;
        if (dma_state.method_count != 0) {
            split_method_end = header.addr + header.size * sizeof(u32);
        }
    }

    if (++dma_pushbuffer_subindex >= command_list_size) {
        // A split method resumes within the submission that split it; ring addresses repeat
        // every frame, so a stale one could otherwise swallow the headers of a later entry.
        for (const auto& parked : parked_methods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 16) {
                LOG_WARNING(HW_GPU,
                            "Submission ended with parked method {:#x} (sub {}, {} words left, "
                            "resume {:#x})",
                            parked.method, parked.subchannel, parked.method_count,
                            parked.resume_address);
            }
        }
        parked_methods.clear();
        dma_pushbuffer.pop_front();
        dma_pushbuffer_subindex = 0;
    } else {
        signal_sync = command_list.command_lists[dma_pushbuffer_subindex].sync && Settings::values.sync_memory_operations.GetValue();
    }

    if (signal_sync) {
        VideoCommon::GpuSplit::Get().Run([this] {
            rasterizer->SignalFence([this]() {
                std::scoped_lock lk(sync_mutex);
                synced = true;
                sync_cv.notify_all();
            });
        }, VideoCommon::DrainSite::GpSyncFence);
    }

    return true;
}

size_t DmaPusher::ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset) {
    for (size_t index = 0; index < commands.size();) {
        // Data word of methods command
        if (dma_state.method_count && dma_state.non_incrementing) {
            auto const& command_header = commands[index]; //must ref (MUltiMethod re)
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            u32 max_write = u32(std::min<std::size_t>(index + dma_state.method_count, commands.size()) - index);
            if (g_nx2_ampere && dma_state.method < non_puller_methods) {
                max_write = 1; // a semaphore acquire may suspend the channel after any word
            }
            CallMultiMethod(&command_header.argument, max_write);
            dma_state.method_count -= max_write;
            dma_state.is_last_call = true;
            index += max_write;
        } else if (dma_state.method_count) {
            // Fast-path batching for consecutive non-execution methods in increasing mode
            if (!dma_state.non_incrementing && !dma_increment_once &&
                dma_state.method >= non_puller_methods &&
                !VideoCommon::GpuSplit::Get().Active() &&
                semaphore_watch == 0) {
                auto subchannel = subchannels[dma_state.subchannel];
                if (subchannel != nullptr) {
                    const size_t available = std::min<size_t>(dma_state.method_count, commands.size() - index);
                    u32 batch_count = 0;
                    while (batch_count < available &&
                           (dma_state.method + batch_count) < subchannel->execution_mask.size() &&
                           !subchannel->execution_mask[dma_state.method + batch_count]) {
                        ++batch_count;
                    }
                    if (batch_count > 1) {
                        for (u32 i = 0; i < batch_count; ++i) {
                            subchannel->method_sink.emplace_back(dma_state.method + i, commands[index + i].argument);
                        }
                        dma_state.method += batch_count;
                        dma_state.method_count -= batch_count;
                        index += batch_count;
                        continue;
                    }
                }
            }

            auto const command_header = commands[index]; //can copy
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            dma_state.is_last_call = dma_state.method_count <= 1;
            CallMethod(command_header.argument);
            dma_state.method += !dma_state.non_incrementing ? 1 : 0;
            dma_state.non_incrementing |= dma_increment_once;
            dma_state.method_count--;
            index++;
        } else {
            auto const command_header = commands[index]; //can copy
            // No command active - this is the first word of a new one
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;
            case SubmissionMode::NonIncreasing:
                SetState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::Inline:
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = u64(-s64(dma_state.dma_get)); // negate to set address as 0
                CallMethod(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::IncreaseOnce:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;
            default:
                break;
            }
            index++;
        }
        if (puller.HasPendingAcquire()) {
            return index;
        }
    }
    return commands.size();
}

// NVN on Switch 2 ends GPFIFO entries in the middle of a method's data and calls another command
// buffer (e.g. a compute dispatch) before the entry that carries the rest, which starts right
// where the first one ended. An entry starting anywhere else is such a call: run it with a clean
// method state and resume the unfinished method when the stream continues at that address.
void DmaPusher::ResolveSplitMethod(GPUVAddr entry_address) {
    const auto continued = std::ranges::find(parked_methods, entry_address,
                                             &ParkedMethod::resume_address);
    std::optional<ParkedMethod> resume;
    if (continued != parked_methods.end()) {
        resume = *continued;
        parked_methods.erase(continued);
    }
    if (dma_state.method_count != 0 && entry_address != split_method_end) {
        if (parked_methods.size() == MaxParkedMethods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 8) {
                const auto& oldest = parked_methods.front();
                LOG_WARNING(HW_GPU,
                            "Dropping parked method {:#x} (sub {}, {} words left, resume {:#x})",
                            oldest.method, oldest.subchannel, oldest.method_count,
                            oldest.resume_address);
            }
            parked_methods.erase(parked_methods.begin());
        }
        parked_methods.push_back(ParkedMethod{
            .method = dma_state.method,
            .subchannel = dma_state.subchannel,
            .method_count = dma_state.method_count,
            .non_incrementing = dma_state.non_incrementing,
            .increment_once = dma_increment_once,
            .resume_address = split_method_end,
        });
        dma_state.method_count = 0;
        dma_increment_once = false;
    }
    if (resume && dma_state.method_count == 0) {
        dma_state.method = resume->method;
        dma_state.subchannel = resume->subchannel;
        dma_state.method_count = resume->method_count;
        dma_state.non_incrementing = resume->non_incrementing;
        dma_increment_once = resume->increment_once;
    }
}

void DmaPusher::SetState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}


void DmaPusher::WatchArgument(u32 method, u32 argument) const {
    const u32 watch = semaphore_watch;
    if ((argument & ~0xfU) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} subchannel {} method {:#x} argument {:#x}",
                 channel_state.bind_id, dma_state.subchannel, method, argument);
    }
}

bool DmaPusher::WantsSegment() const {
    // Only macro parameters and compute uploads read the engine's current_dma_segment.
    const auto engine = subchannel_type[dma_state.subchannel];
    return engine == Engines::EngineTypes::KeplerCompute ||
           (engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart);
}

void DmaPusher::CallMethod(u32 argument) {
    if (semaphore_watch != 0) [[unlikely]] {
        WatchArgument(dma_state.method, argument);
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallPullerMethod(*this, Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        const auto engine_type = subchannel_type[dma_state.subchannel];
        ShadowEngineMethod(engine_type, dma_state.method, argument);
        const bool writer = split.Active() &&
                            IsMemoryWriter(*subchannel, engine_type, dma_state.method);
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter();
                if (engine_type == Engines::EngineTypes::MaxwellDMA) {
                    g_last_copy_writer = split.Position();
                }
            }
        };
        if (split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        maxwell3d->FrontCallMethod(system, split, dma_state.method, argument,
                                                   dma_state.is_last_call);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMethod(system, split, dma_state.method, argument,
                                                 dma_state.is_last_call, &segment);
                        return;
                    }
                }
            }
            split.Method(subchannel, dma_state.method, argument, dma_state.is_last_call,
                         WantsSegment() ? &segment : nullptr,
                         EngineWrites(*subchannel, engine_type, dma_state.method));
            split.TagUnplaced(VideoCommon::UnplacedKind::Engine); // TEST
            NoteEngineWrites(engine_type, dma_state.method);
            return;
        }
        if (!subchannel->execution_mask[dma_state.method]) {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
        } else {
            VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
            subchannel->ConsumeSink(system);
            subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
            subchannel->CallMethod(system, dma_state.method, argument, dma_state.is_last_call);
        }
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) {
    if (semaphore_watch != 0) [[unlikely]] {
        for (u32 i = 0; i < num_methods; ++i) {
            WatchArgument(dma_state.method + (dma_state.non_incrementing ? 0 : i), base_start[i]);
        }
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallMultiMethod(*this, dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        const auto engine_type = subchannel_type[dma_state.subchannel];
        const bool writer = split.Active() &&
                            IsMemoryWriter(*subchannel, engine_type, dma_state.method);
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter();
                if (engine_type == Engines::EngineTypes::MaxwellDMA) {
                    g_last_copy_writer = split.Position();
                }
            }
        };
        if (split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        maxwell3d->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                        num_methods, dma_state.method_count);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                      num_methods, dma_state.method_count, &segment);
                        return;
                    }
                }
            }
            split.MultiMethod(subchannel, dma_state.method, base_start, num_methods,
                              dma_state.method_count, WantsSegment() ? &segment : nullptr,
                              EngineWrites(*subchannel, engine_type, dma_state.method));
            split.TagUnplaced(VideoCommon::UnplacedKind::Engine); // TEST
            // The back runs them one by one: a copy engine launch each.
            const bool launches = engine_type == Engines::EngineTypes::MaxwellDMA &&
                                  dma_state.method == Engines::MaxwellDMA::LaunchDmaMethod;
            for (u32 i = 0; i < num_methods; ++i) {
                ShadowEngineMethod(engine_type, dma_state.method, base_start[i]);
                if (launches) {
                    NoteEngineWrites(engine_type, dma_state.method);
                }
            }
            if (!launches && num_methods != 0) {
                NoteEngineWrites(engine_type, dma_state.method);
            }
            return;
        }
        for (u32 i = 0; i < num_methods; ++i) {
            ShadowEngineMethod(engine_type, dma_state.method, base_start[i]);
        }
        VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
        subchannel->ConsumeSink(system);
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(system, dma_state.method, base_start, num_methods, dma_state.method_count);
    }
}

void DmaPusher::ShadowEngineMethod(Engines::EngineTypes type, u32 method, u32 argument) {
    // As the engines' CallMethod and ConsumeSink store them.
    if (type == Engines::EngineTypes::MaxwellDMA) {
        if (method < dma_registers.size()) {
            dma_registers[method] = argument;
        }
    } else if (type == Engines::EngineTypes::KeplerMemory) {
        if (method < inline_to_memory_registers.size()) {
            inline_to_memory_registers[method] = argument;
        }
    }
}

void DmaPusher::NoteEngineWrites(Engines::EngineTypes type, u32 method) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (type == Engines::EngineTypes::KeplerMemory) {
        if (method != KEPLERMEMORY_REG_INDEX(data)) {
            return;
        }
        // Upload::State::ProcessData, of the upload KeplerMemory's exec started.
        Engines::Upload::Registers upload;
        std::memcpy(&upload, &inline_to_memory_registers[KEPLERMEMORY_REG_INDEX(upload)],
                    sizeof(upload));
        const bool linear = (inline_to_memory_registers[KEPLERMEMORY_REG_INDEX(exec)] & 1) != 0;
        const auto range = Engines::Upload::UploadedRange(upload, linear);
        split.NoteWrite(memory_manager, range.address, range.size);
        return;
    }
    using Engines::MaxwellDMA;
    if (type != Engines::EngineTypes::MaxwellDMA || method != MaxwellDMA::LaunchDmaMethod) {
        return;
    }
    const auto writes = MaxwellDMA::WritesOfLaunch(dma_registers);
    split.NoteWrite(memory_manager, writes.destination.address, writes.destination.size);
    if (writes.semaphore.size == 0) {
        return;
    }
    // As MaxwellDMA::ReleaseSemaphore writes it.
    split.NoteWrite(memory_manager, writes.semaphore.address, writes.semaphore.size);
    const auto device = memory_manager.GpuToCpuAddress(writes.semaphore.address);
    if (!device) {
        return;
    }
    // Fence-flagged reports: they land once their fence ran.
    constexpr auto landing = VideoCommon::GpuSplit::Landing::Fenced;
    split.NoteSemaphoreWrite(*device, dma_registers[MaxwellDMA::SemaphoreMethod + 2], true, false,
                             landing);
    VideoCommon::Nx2AcquireStats::NoteRelease(*device,
                                              VideoCommon::Nx2AcquireStats::Releaser::Copy);
    if (writes.semaphore.size > sizeof(u32)) {
        // The payload as 64 bits (a timestamp follows).
        split.NoteSemaphoreWrite(*device + 4, 0, true, false, landing);
    }
}

void DmaPusher::RefreshSnapshot(CommandList& list, u64 now, size_t first_entry) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Resnapshot}; // TEST
    list.snapshot_suspensions = now;
    for (size_t i = first_entry; i < list.command_lists.size() && i < list.submitted_offsets.size();
         ++i) {
        const CommandListHeader& entry = list.command_lists[i];
        const u32 offset = list.submitted_offsets[i];
        if (offset + entry.size > list.submitted_words.size()) {
            continue;
        }
        memory_manager.ReadBlockUnsafe(entry.addr, list.submitted_words.data() + offset,
                                       entry.size * sizeof(u32));
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::ResnapshotWords,
                                     entry.size); // TEST
    }
    list.deferred_entries.clear();
}

void DmaPusher::RefreshDeferredEntries(CommandList& list) {
    for (const u32 i : list.deferred_entries) {
        if (i >= list.command_lists.size() || i >= list.submitted_offsets.size()) {
            continue;
        }
        const CommandListHeader& entry = list.command_lists[i];
        const u32 offset = list.submitted_offsets[i];
        if (offset + entry.size > list.submitted_words.size()) {
            continue;
        }
        memory_manager.ReadBlockUnsafe(entry.addr, list.submitted_words.data() + offset,
                                       entry.size * sizeof(u32));
    }
    list.deferred_entries.clear();
}

bool DmaPusher::IsGpuWrittenCommands(GPUVAddr address, size_t size) {
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return false;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            return true;
        }
    }
    return false;
}

void DmaPusher::CheckUnwaitedWriters(GPUVAddr address, size_t size) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active() || !split.WritersPending()) {
        return;
    }
    // The list did not wait for compute or copy work sent after the last sync. At the list's
    // place, the back has run that work: did it make the list's pages GPU-written?
    VideoCommon::Nx2DrainCheck::NoteListUnwaited();
    split.Defer(
        [memory = &memory_manager, address, size] {
            bool written = false;
            for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size;
                 page += 0x1000) {
                const auto device = memory->GpuToCpuAddress(page);
                written = written || (device && Tegra::Nx2IsGpuWritten(*device));
            }
            VideoCommon::Nx2DrainCheck::NoteListChecked(written);
        },
        VideoCommon::GpuSplit::Writes::Noted);
}

bool DmaPusher::ReadGpuWrittenCommands(GPUVAddr address, u32 count, bool* pending) {
    // The pages are marked as command list pages: from the next write on, the buffer cache copies
    // them to host memory right after the dispatch or copy that writes them and submits. The
    // front then waits only for the back to run that work and for the GPU to finish the copy, not
    // for everything queued and a cache flush (FlushGpuWrittenCommands: ~400 GPU round trips a
    // second in a Mario Kart World race, during which the back had nothing to do). Lists on words
    // no GPU work wrote since FlushGpuWrittenCommands flushed the page, or the host wrote them,
    // are read from guest memory and wait for no copy (BufferCache::Nx2ReadListCopy).
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || count == 0 ||
        rasterizer == nullptr) {
        return false;
    }
    const GPUVAddr end = address + u64{count} * sizeof(u32);
    bool written = false;
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < end; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            Tegra::Nx2MarkCommandListPage(*device);
            written = true;
        }
    }
    if (!written) {
        return false;
    }
    // As for lists on pages no GPU work wrote (Step): the dispatch or copy that rewrites the
    // list is ordered before the fetch only by a host-level wait between them, so the front waits
    // for the work before the last one, not for compute and copy work sent after it.
    auto& split = VideoCommon::GpuSplit::Get();
    // Resuming after the copy's tick: the back ran the list's writers before the suspension
    // (this wait ran then), and waiting again would wait for the other channels' work queued
    // meanwhile too.
    const bool resumed = std::exchange(resumed_list_read, false);
    if (split.Active() && !resumed) {
        split.WaitForSyncedWriters(VideoCommon::DrainSite::ListCopies);
    }
    bool past_sync = !split.Active();
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    if (Nx2ListStats::Any()) [[unlikely]] {
        Nx2ListStats::BeginList(count);
    }
    command_headers.resize_destructive(count);
    u8* const out = reinterpret_cast<u8*>(command_headers.data());
    for (GPUVAddr chunk = address; chunk < end;) {
        const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
        const std::span<u8> part{out + (chunk - address), static_cast<size_t>(chunk_end - chunk)};
        const auto device = memory_manager.GpuToCpuAddress(chunk);
        if (!device) {
            return false;
        }
        GpuWrittenList where = GpuWrittenList::Synced;
        if (Tegra::Nx2IsGpuWritten(*device)) {
            const bool stats = Nx2ListStats::g_enabled && split.Active();
            const u64 backlog = stats ? split.Backlog() : 0;
            const u64 unpublished = stats ? split.Unpublished() : 0;
            u64 tick = 0;
            u64* const pending_tick = pending != nullptr ? &tick : nullptr;
            where = rasterizer->ReadGpuWrittenCommandList(*device, part, past_sync, pending_tick);
            const bool after_sync = where == GpuWrittenList::NeedsSync;
            if (where == GpuWrittenList::NeedsSync) {
                const auto wait_start = Nx2CmdPageDiag::g_enabled
                                            ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
                const bool waited = split.WaitForSync();
                if (Nx2CmdPageDiag::g_enabled && waited) {
                    Nx2CmdPageDiag::NoteSyncWait(std::chrono::steady_clock::now() - wait_start);
                }
                past_sync = true;
                where = rasterizer->ReadGpuWrittenCommandList(*device, part, true, pending_tick);
            }
            if (where == GpuWrittenList::Pending) {
                // The copy is still on the GPU: the channel waits for it without blocking the
                // others (Step parses this entry again once IsGpuTickDone).
                pending_list_tick = tick;
                *pending = true;
                return false;
            }
            if (Nx2CmdPageDiag::g_enabled) {
                auto& diag = Nx2CmdPageDiag::Get();
                const u64 page = *device & ~u64{0xfff};
                if (where == GpuWrittenList::Copied) {
                    diag.reads_copied.fetch_add(1, std::memory_order_relaxed);
                } else if (where == GpuWrittenList::Synced) {
                    diag.reads_synced.fetch_add(1, std::memory_order_relaxed);
                    if (split.Active() && Nx2CmdPageDiag::IsDrawPage(page)) {
                        // The back, at this list's place: did work sent before it write the page?
                        split.Defer(
                            [page, begin = static_cast<u32>(*device - page),
                             end = static_cast<u32>(*device - page + part.size())] {
                                Nx2CmdPageDiag::CheckMissedWrite(page, begin, end);
                            },
                            VideoCommon::GpuSplit::Writes::Anywhere,
                            VideoCommon::GpuSplit::FastDrawNeutral);
                    }
                } else {
                    diag.reads_unknown.fetch_add(1, std::memory_order_relaxed);
                    if (Nx2CmdPageDiag::IsDrawPage(page)) {
                        diag.reads_unknown_draw.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            if (Nx2ListStats::Any()) [[unlikely]] {
                NoteListRead(memory_manager, channel_state.bind_id, chunk, *device, part, where,
                             after_sync, backlog, unpublished);
            }
        }
        if (where == GpuWrittenList::Unknown) {
            return false;
        }
        if (where == GpuWrittenList::Synced) {
            memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
        }
        chunk = chunk_end;
    }
    if (Nx2ListStats::g_verify) [[unlikely]] {
        VerifyGpuWrittenCommands(address, count);
    }
    return true;
}

bool DmaPusher::TakeGpuWrittenListWords(const CommandListHeader& header) {
    // Ocarina of Time's compute culling writes the group counts of a later dispatch as the
    // parameters of compute macro 0xe74, which only stores them in the data RAM for 0xe7a and
    // 0xe76 to upload and launch with. The front need not wait for the culling to run and its
    // copy to come back: the macros run on placeholders the host GPU's copy of the words takes
    // the place of (Nx2ListIndirect, KeplerCompute::FrontTakeListWords).
    auto& split = VideoCommon::GpuSplit::Get();
    using Kept = Nx2ListIndirect::Stats::Kept; // TEST
    if (Nx2ListIndirect::Stats::Enabled()) [[unlikely]] {
        Nx2ListIndirect::Stats::MaybeLog();
    }
    if (!Nx2ListIndirect::g_enabled || rasterizer == nullptr || command_resume_offset != 0 ||
        !split.Active() || !split.FrontMacros()) {
        Nx2ListIndirect::Stats::NoteKept(Kept::Off);
        return false;
    }
    const u32 count = static_cast<u32>(header.size);
    // All of a macro call's parameters, from its first: its header ended the entry before.
    if (count == 0 || count > Nx2ListIndirect::SlotWords || dma_state.method_count != count ||
        dma_state.method < MacroRegistersStart || (dma_state.method & 1) != 0 ||
        subchannels[dma_state.subchannel] == nullptr ||
        subchannel_type[dma_state.subchannel] != Engines::EngineTypes::KeplerCompute) {
        Nx2ListIndirect::Stats::NoteKept(Kept::NotCall);
        return false;
    }
    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannels[dma_state.subchannel]);
    if (!compute->FrontRunsMacros() || !rasterizer->Nx2ListSlotsSupported()) {
        Nx2ListIndirect::Stats::NoteKept(Kept::Off);
        return false;
    }
    // On one device page, as the back finds it: its mappings are the front's once it ran the
    // last change sent.
    const u64 bytes = u64{count} * sizeof(u32);
    const auto device = memory_manager.GpuToCpuAddress(dma_state.dma_get);
    const auto last = memory_manager.GpuToCpuAddress(dma_state.dma_get + bytes - 1);
    if (!device || !last || *last != *device + bytes - 1 || !split.Ran(split.LastMappingChange())) {
        Nx2ListIndirect::Stats::NoteKept(Kept::Placement);
        return false;
    }
    const std::optional<u32> slot = rasterizer->Nx2AcquireListSlot();
    if (!slot) {
        Nx2ListIndirect::Stats::NoteKept(Kept::NoSlot);
        return false;
    }
    if (!compute->FrontTakeListWords(split, dma_state.method, count, *slot)) {
        rasterizer->Nx2ReleaseListSlot(*slot); // Nothing sent with it
        Nx2ListIndirect::Stats::NoteKept(Kept::Refused);
        return false;
    }
    Nx2ListIndirect::Stats::NoteTaken(dma_state.method, count);
    // TEST (EDEN_NX2_LIST_INDIRECT_CHECK): the words as the old path reads them (waiting for
    // their writers), for the back to compare what the host GPU uses of the slot with.
    const bool check = Nx2ListIndirect::Stats::g_check;
    std::array<u32, Nx2ListIndirect::SlotWords> expected{};
    if (check) [[unlikely]] {
        if (ReadGpuWrittenCommands(dma_state.dma_get, count)) {
            for (u32 k = 0; k < count; ++k) {
                expected[k] = command_headers[k].argument;
            }
        } else {
            FlushGpuWrittenCommands(dma_state.dma_get, bytes);
            memory_manager.ReadBlockUnsafe(dma_state.dma_get, expected.data(), bytes);
        }
    }
    // At the list's place: after the work sent before it, the GPU fetched the list there.
    split.Defer(
        [rasterizer_ = rasterizer, device_ = *device, count, slot_ = *slot, check, expected] {
            rasterizer_->Nx2SnapshotListWords(device_, count, slot_);
            if (check) {
                rasterizer_->Nx2CheckListSlot(slot_, std::span{expected}.first(count));
            }
        },
        VideoCommon::GpuSplit::Writes::Noted);
    command_headers.resize_destructive(count);
    for (u32 k = 0; k < count; ++k) {
        command_headers[k].argument = Nx2ListIndirect::Placeholder(*slot, k);
    }
    return true;
}

void DmaPusher::VerifyGpuWrittenCommands(GPUVAddr address, u32 count) {
    // The list again as the front read lists before it tracked the words GPU work wrote: after
    // every dispatch and copy sent so far, from the pages' whole copies, or drained and flushed.
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    const size_t bytes = size_t{count} * sizeof(u32);
    std::vector<u32> parsed(count);
    std::memcpy(parsed.data(), command_headers.data(), bytes);
    auto& split = VideoCommon::GpuSplit::Get();
    if (split.Active()) {
        split.WaitForWriters();
    }
    bool past_sync = !split.Active();
    std::vector<u32> reference(count);
    bool reference_drained = false;
    {
        const Nx2ListStats::ReferenceScope reference_read;
        u8* const out = reinterpret_cast<u8*>(reference.data());
        const GPUVAddr end = address + bytes;
        for (GPUVAddr chunk = address; chunk < end && !reference_drained;) {
            const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
            const std::span<u8> part{out + (chunk - address),
                                     static_cast<size_t>(chunk_end - chunk)};
            const auto device = memory_manager.GpuToCpuAddress(chunk);
            GpuWrittenList where = device ? GpuWrittenList::Synced : GpuWrittenList::Unknown;
            if (device && Tegra::Nx2IsGpuWritten(*device)) {
                where = rasterizer->ReadGpuWrittenCommandList(*device, part, past_sync);
                if (where == GpuWrittenList::NeedsSync) {
                    split.WaitForSync();
                    past_sync = true;
                    where = rasterizer->ReadGpuWrittenCommandList(*device, part, true);
                }
            }
            if (where == GpuWrittenList::Synced) {
                memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
            }
            reference_drained = where == GpuWrittenList::Unknown;
            chunk = chunk_end;
        }
    }
    const auto read_drained = [&](std::vector<u32>& words) {
        FlushGpuWrittenCommands(address, bytes);
        memory_manager.ReadBlockUnsafe(address, words.data(), bytes);
    };
    if (reference_drained) {
        read_drained(reference);
    }
    std::vector<u32> drained;
    if (!reference_drained && reference != parsed) {
        drained.resize(count);
        read_drained(drained);
    }
    Nx2ListStats::NoteVerify(address, parsed, reference, drained, reference_drained);
    // The front parses the words read last.
    std::memcpy(command_headers.data(), drained.empty() ? reference.data() : drained.data(), bytes);
}

bool DmaPusher::FlushGpuWrittenCommands(GPUVAddr address, size_t size) {
    if (!g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return false;
    }
    if (!IsGpuWrittenCommands(address, size)) {
        return false;
    }
    // The caches belong to the split's back: it first runs what the front queued (the dispatch
    // that writes these words among it). The whole pages are flushed: their next lists read
    // guest memory until GPU work writes them again.
    const bool diag = Nx2CmdPageDiag::g_enabled;
    const auto diag_start =
        diag ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<u32> diag_before;
    if (diag) {
        diag_before.resize(size / sizeof(u32));
        memory_manager.ReadBlockUnsafe(address, diag_before.data(), diag_before.size() * sizeof(u32));
    }
    VideoCommon::GpuSplit::Get().Drain(VideoCommon::DrainSite::ListFlush);
    const GPUVAddr first_page = address & ~GPUVAddr{0xfff};
    const GPUVAddr end_page = (address + size + 0xfff) & ~GPUVAddr{0xfff};
    const bool dirty = memory_manager.IsMemoryDirty(first_page, end_page - first_page,
                                                    VideoCommon::CacheType::BufferCache);
    if (rasterizer != nullptr) {
        std::vector<DAddr> pages;
        for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
            if (const auto device = memory_manager.GpuToCpuAddress(page)) {
                pages.push_back(*device);
            }
        }
        VideoCommon::GpuSplit::Get().Run(
            [rasterizer = rasterizer, pages = std::span<const DAddr>{pages}] {
                rasterizer->FlushGpuWrittenCommandListPages(pages);
            },
            VideoCommon::DrainSite::ListFlush);
    } else if (dirty) {
        VideoCommon::GpuSplit::Get().Run(
            [memory = &memory_manager, first_page, size = end_page - first_page] {
                memory->FlushRegion(first_page, size, VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::ListFlush);
    }
    if (diag) {
        auto kind = Nx2CmdPageDiag::Kind::NoWrite;
        Nx2CmdPageDiag::PageView first_view;
        u64 first_device{};
        for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            if (!device) {
                continue;
            }
            const u32 begin = static_cast<u32>((std::max)(address, page) - page);
            const u32 end = static_cast<u32>((std::min)(address + size, page + 0x1000) - page);
            const auto view = Nx2CmdPageDiag::TakePage(*device, begin, end);
            if (page == first_page) {
                first_view = view;
                first_device = *device;
            }
            kind = (std::max)(kind, view.kind);
        }
        std::vector<u32> after(diag_before.size());
        memory_manager.ReadBlockUnsafe(address, after.data(), after.size() * sizeof(u32));
        u64 changed{};
        for (size_t i = 0; i < after.size(); ++i) {
            changed += after[i] != diag_before[i] ? 1 : 0;
        }
        const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - diag_start)
                                            .count());
        Nx2CmdPageDiag::RecordSlow(kind, changed, dirty, ns, address, first_device,
                                   static_cast<u32>(address - first_page),
                                   static_cast<u32>((std::min)(address + size, first_page + 0x1000) -
                                                    first_page),
                                   first_view);
    }
    if (rasterizer != nullptr) {
        for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
            if (const auto device = memory_manager.GpuToCpuAddress(page);
                device && Tegra::Nx2IsCommandListPage(*device)) {
                rasterizer->NoteGpuWrittenCommandListSynced(*device);
            }
        }
    }
    return true;
}

void DmaPusher::ReportUnboundSubchannel() {
    // Methods sent to a subchannel with no engine (no SetObject yet) are dropped.
    static std::atomic<u32> reported{};
    if (reported.fetch_add(1) < 8) {
        LOG_WARNING(HW_GPU, "Dropping method {:#x} on unbound subchannel {}", dma_state.method,
                    dma_state.subchannel);
    }
}

void DmaPusher::CheckSubmitHash(const CommandListHeader& header, u64 submit_hash) {
    std::vector<u32> words(header.size);
    memory_manager.ReadBlockUnsafe(header.addr, words.data(), words.size() * sizeof(u32));
    static std::atomic<u64> checked{};
    static std::atomic<u64> changed{};
    ++checked;
    if (HashPushbufferWords(words) != submit_hash) {
        if (changed.fetch_add(1) < 16) {
            LOG_WARNING(HW_GPU, "pushbuffer entry {:#x}+{:#x} changed between submit and execution",
                        header.addr.Value(), header.size.Value());
        }
    }
    if ((checked % 20000) == 0) {
        LOG_WARNING(HW_GPU, "pushbuffer check: {} of {} entries changed after submit",
                    changed.load(), checked.load());
    }
}

void DmaPusher::RecordParsedEntry(const CommandListHeader& header,
                                  std::span<const CommandHeader> commands) {
    if (!record_parsed_entries) {
        return;
    }
    u64 hash = 0xcbf29ce484222325ULL;
    for (const CommandHeader& command : commands) {
        hash = (hash ^ command.argument) * 0x100000001b3ULL;
    }
    parsed_entries[parsed_entry_count++ % parsed_entries.size()] = {header.addr, static_cast<u32>(header.size), hash};
}

void DmaPusher::DumpCurrentEntry(const char* reason) {
    // Entries whose memory changed after they were parsed (the CPU rewrote them meanwhile).
    for (size_t i = 0; i < std::min(parsed_entry_count, parsed_entries.size()); ++i) {
        const auto& entry = parsed_entries[(parsed_entry_count - 1 - i) % parsed_entries.size()];
        std::vector<u32> words(entry.size);
        memory_manager.ReadBlockUnsafe(entry.address, words.data(), words.size() * sizeof(u32));
        u64 hash = 0xcbf29ce484222325ULL;
        for (const u32 word : words) {
            hash = (hash ^ word) * 0x100000001b3ULL;
        }
        LOG_WARNING(HW_GPU, "  parsed -{}: {:#x}+{:#x} {}", i, entry.address, entry.size,
                    hash == entry.hash ? "unchanged" : "CHANGED since parsed");
    }
    // EDEN_NX2_PB_DUMP=<file>: also write the whole command list (u64 address, u32 size and the
    // current words of every entry, then the current entry index) for offline replay.
    if (static const char* const path = std::getenv("EDEN_NX2_PB_DUMP");
        path != nullptr && !dma_pushbuffer.empty()) {
        if (std::FILE* file = std::fopen(path, "wb")) {
            const auto& lists = dma_pushbuffer.front().command_lists;
            const u64 current = dma_pushbuffer_subindex;
            const u64 count = lists.size();
            std::fwrite(&count, sizeof(count), 1, file);
            std::fwrite(&current, sizeof(current), 1, file);
            for (const CommandListHeader& header : lists) {
                const u64 address = header.addr;
                const u32 size = static_cast<u32>(header.size);
                std::vector<u32> words(size);
                memory_manager.ReadBlockUnsafe(address, words.data(), words.size() * sizeof(u32));
                std::fwrite(&address, sizeof(address), 1, file);
                std::fwrite(&size, sizeof(size), 1, file);
                std::fwrite(words.data(), sizeof(u32), words.size(), file);
            }
            std::fclose(file);
        }
    }
    LOG_WARNING(HW_GPU, "pushbuffer dump ({}): method={:#x} sub={} count={} non_inc={} get={:#x}",
                reason, dma_state.method, dma_state.subchannel, dma_state.method_count,
                dma_state.non_incrementing, dma_state.dma_get);
    if (dma_pushbuffer.empty()) {
        return;
    }
    const auto& lists = dma_pushbuffer.front().command_lists;
    const size_t first = dma_pushbuffer_subindex >= 2 ? dma_pushbuffer_subindex - 2 : 0;
    for (size_t i = first; i <= dma_pushbuffer_subindex && i < lists.size(); ++i) {
        const CommandListHeader& header = lists[i];
        LOG_WARNING(HW_GPU, "  entry {} of {}: addr={:#x} size={:#x}", i, lists.size(),
                    header.addr.Value(), header.size.Value());
        for (u64 word = 0; word < std::min<u64>(header.size, 0x80); word += 8) {
            std::string gpu_line;
            std::string cpu_line;
            for (u64 j = word; j < std::min<u64>(header.size, word + 8); ++j) {
                const GPUVAddr address = header.addr + j * sizeof(u32);
                const u32 gpu_value = memory_manager.Read<u32>(address);
                const u32 cpu_value = system.ApplicationMemory().Read32(address);
                gpu_line += fmt::format(" {:08x}", gpu_value);
                cpu_line += gpu_value == cpu_value ? std::string(" ........")
                                                   : fmt::format(" {:08x}", cpu_value);
            }
            LOG_WARNING(HW_GPU, "    +{:03x}:{}", word * 4, gpu_line);
            LOG_WARNING(HW_GPU, "    cpu :{}", cpu_line);
        }
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

} // namespace Tegra
