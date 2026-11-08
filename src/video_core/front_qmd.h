// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "video_core/gpu_split.h"
#include "video_core/native_qmd.h"
#include "video_core/nx2_drain_check.h"

namespace VideoCommon {

/// The QMD scheduling of every channel (Tegra::NativeQmdScheduler), run on the split GPU thread's
/// front as its back will run it, on a table of its own: what the launches release is known when
/// they are sent, for the front's view of the semaphore words (GpuSplit::NoteSemaphoreWrite).
///
/// The table is the back's from the start: an engine's front takes it over with the back drained
/// when it takes the engine's macros over (TakeOver). It stays the back's as long as the front
/// reads the descriptors the back will read. The front reads one when it sends a signal, the back
/// when it runs it, both from guest memory without the GPU caches. The front takes a descriptor:
/// - from the signal's own inline QMD, which the back writes right before;
/// - from memory, once the back ran every write sent there (GpuSplit::MemoryWritesTarget: the
///   writes placed in its granule and the unplaced ones), on pages GPU work never wrote
///   (Nx2MayBeGpuWritten);
/// - from the inline QMD a LOAD_INLINE_QMD_DATA still queued writes there, when that is the last
///   write sent there (NoteInlineWrite).
/// No change of the GPU's mappings may be queued either (GpuSplit::LastMappingChange). Memory holds
/// what the back finds there only while the back has not run the signal's entry yet: the front
/// sends and values it under a GpuSplit::PublishHold.
///
/// Where the back has yet to run a write the front can't read past (into a QMD the signal reads,
/// or anywhere, or to the mappings), the signal waits, with what it changed in the table taken
/// back (NativeQmdScheduler::BeginAttempt), and every signal sent after it waits behind it: the
/// table stays the back's as of the signals valued. The front values the waiting ones in order
/// once the back ran what the first waits for (CatchUp). Late, the back may have run the signal
/// and what followed it, so memory holds what the back read only where nothing sent after the
/// signal writes the QMD, and no release of the waiting signals writes a QMD read late. Their
/// releases are placed at their signals' positions, not noted. Where the front can't read what
/// the back read at all (a QMD on a GPU-written page, written after the signal, released into,
/// or the mappings changed after it), it loses the table: it stops valuing, and takes the back's
/// table over the next time the back has run everything sent, between commands or at a QMD
/// method (MaybeTakeOver).
///
/// `Memory`, the channel's GPU address space as the back reads it, has:
/// - bool IsMapped(GPUVAddr, u64), void Read(GPUVAddr, void*, u64), u32 ReadWord(GPUVAddr) (as
///   MemoryManager::Read<u32>), bool InSyncpointShim(GPUVAddr, u64);
/// - std::optional<u64> Device(GPUVAddr), bool GpuWritten(u64 device, u64 size);
/// - u64 WritesTarget(const GpuSplit&, GPUVAddr, u64), void NoteWrite(GpuSplit&, GPUVAddr, u64),
///   void NoteWriteAt(GpuSplit&, GPUVAddr, u64 size, u64 position);
/// - const void* Id(): the address space's, the same for every Memory of it.
class FrontQmdView {
public:
    using Descriptor = Tegra::NativeQmdScheduler::Descriptor;

    /// The view the compute engines of every channel feed: they share the back's table.
    static FrontQmdView& Get() {
        static FrontQmdView view;
        return view;
    }

    /// The back ran everything sent and runs nothing (the split drained, or it waits for work):
    /// takes over the table `back` shares with the other engines. `where`: TEST.
    void TakeOver(const Tegra::NativeQmdScheduler& back, Nx2AcquireStats::QmdTakeOver where) {
        const auto start = std::chrono::steady_clock::now(); // TEST
        scheduler.CopyStates(back);
        synced = true;
        Nx2AcquireStats::NoteQmdTakeOverDropped(waiting.size()); // TEST
        ClearWaiting();
        Nx2AcquireStats::NoteQmdSync(where, std::chrono::steady_clock::now() - start);
    }

    /// The front, between commands and before QMD methods: takes the back's table over if the
    /// view lost it or signals wait, and the back has run everything sent.
    void MaybeTakeOver(const GpuSplit& split, const Tegra::NativeQmdScheduler& back,
                       Nx2AcquireStats::QmdTakeOver where) {
        if (!Current() && split.Ran(split.Position())) [[unlikely]] {
            TakeOver(back, where);
        }
    }
    void MaybeTakeOver(const GpuSplit& split, Nx2AcquireStats::QmdTakeOver where) {
        MaybeTakeOver(split, back_table, where);
    }

    /// TEST (EDEN_NX2_DRAIN_CHECK): the releases noted since the last call, the descriptors read
    /// for them, and whether every signal since was valued; every 512th time while the view is
    /// current, and after it valued signals late, a copy of its table; the position of the entry
    /// whose signals wait, from the `first_waiting`-th on; the signals valued late since, with
    /// what they read; nothing if no signal was sent since.
    struct Checks {
        std::vector<Nx2DrainCheck::QmdRelease> releases;
        std::vector<Nx2DrainCheck::QmdRead> reads;
        std::shared_ptr<const Tegra::NativeQmdScheduler> table;
        bool valued;
        bool table_after_late;
        u64 wait_position;
        u32 first_waiting;
        std::vector<Nx2DrainCheck::LateQmdRead> late;
    };
    [[nodiscard]] std::optional<Checks> TakeChecks() {
        if (!checks_pending) {
            return std::nullopt;
        }
        checks_pending = false;
        Checks taken{std::move(checks),
                     std::move(check_reads),
                     nullptr,
                     !checks_unvalued,
                     false,
                     check_wait_position,
                     check_first_waiting,
                     std::move(late_checks)};
        if (taken.valued && Current() && (table_check_due || ++table_checks % 512 == 0)) {
            auto copy =
                std::make_shared<Tegra::NativeQmdScheduler>(Tegra::NativeQmdScheduler::OwnTable{});
            copy->CopyStates(scheduler);
            taken.table = std::move(copy);
            taken.table_after_late = table_check_due;
            table_check_due = false;
        }
        checks.clear();
        check_reads.clear();
        late_checks.clear();
        checks_unvalued = false;
        check_wait_position = 0;
        check_first_waiting = 0;
        return taken;
    }

    /// The engines schedule QMDs the front does not see (the split stopped).
    void Lose() {
        synced = false;
        inline_writes_count = 0;
        ClearWaiting();
    }

    [[nodiscard]] bool Synced() const {
        return synced;
    }
    /// Whether the table is the back's as of everything sent: synced, and no signal waits.
    [[nodiscard]] bool Current() const {
        return synced && waiting.empty();
    }
    /// TEST: whether the view's table holds the QMD states `other` does.
    [[nodiscard]] bool SameTable(const Tegra::NativeQmdScheduler& other) const {
        return scheduler.SameStates(other);
    }

    /// LOAD_INLINE_QMD_DATA was just sent: running it, the back writes `qmd` to the QMD at
    /// `device`. Until then the front reads it from here (ReadDescriptor).
    void NoteInlineWrite(const GpuSplit& split, u64 device, const Descriptor& qmd) {
        InlineWrite& written = inline_writes[inline_writes_count % inline_writes.size()];
        ++inline_writes_count;
        written.device = device;
        written.position = split.Position();
        written.qmd = qmd;
        written.overwritten = false;
    }

    /// The scheduling of PCAS action `argument` on the QMD at `address` that the back runs at the
    /// entry just sent (KeplerCompute::ProcessNativeMethod); `inline_qmd`: the QMD it writes there
    /// right before. Notes what the launches it runs release, or has the signal wait. The entry
    /// must still be unpublished (GpuSplit::PublishHold): the back must not have run it.
    template <typename Memory>
    void Signal(GpuSplit& split, const Memory& memory, GPUVAddr address, u32 argument,
                const Descriptor* inline_qmd) {
        split.NoteCacheWrite(); // The dispatches' shader stores
        using Lost = Nx2AcquireStats::QmdLost;
        checks_pending = Nx2DrainCheck::g_enabled; // TEST
        const u64 position = split.Position();
        if (checks_pending) {
            // TEST: the back must not have run the entry yet, the n-th signal of its entry.
            if (split.Unpublished() == 0) [[unlikely]] {
                Nx2DrainCheck::NotePublishedEarly();
            }
            entry_signals = position != entry_position ? 0 : entry_signals + 1;
            entry_position = position;
        }
        if (synced) {
            CatchUp(split, memory);
        }
        if (!synced) {
            Unvalued(split, Lost::NotSynced);
            return;
        }
        if (!waiting.empty()) {
            // Behind a signal still waiting.
            Wait(split, position, memory.Id(), address, argument, inline_qmd, std::nullopt);
            return;
        }
        const Outcome outcome =
            Attempt(split, memory, position, address, argument, inline_qmd, false);
        if (outcome.valued) {
            Nx2AcquireStats::NoteQmdSignal(std::nullopt);
            return;
        }
        if (outcome.retry_after) {
            retry_after = *outcome.retry_after;
            if (Nx2AcquireStats::g_enabled && outcome.lost == Lost::Pending) [[unlikely]] {
                NoteWaitWriter(split, memory, position, outcome); // TEST
            }
            Wait(split, position, memory.Id(), address, argument, inline_qmd, outcome.lost);
            return;
        }
        synced = false;
        Unvalued(split, outcome.lost);
    }

private:
    struct Release {
        GPUVAddr address;
        u32 payload;
        u32 flags;
    };

    /// An inline QMD a LOAD_INLINE_QMD_DATA sent writes (NoteInlineWrite).
    struct InlineWrite {
        u64 device{};
        u64 position{};
        Descriptor qmd{};
        bool overwritten{}; ///< A release of the same entry writes into it after
    };

    /// A signal sent at `position` that waits to be valued (`inline_qmd` if `has_inline`).
    struct WaitingSignal {
        u64 position;
        const void* memory; ///< Memory::Id of the address space it reads
        GPUVAddr address;
        u32 argument;
        bool has_inline;
        Descriptor inline_qmd;
        u32 index; ///< TEST: of the signals of its entry
    };

    /// How an attempt to value a signal ended: valued, or taken back, to try again once the back
    /// ran `retry_after`, or for good (the view loses its table).
    struct Outcome {
        bool valued;
        std::optional<u64> retry_after;
        Nx2AcquireStats::QmdLost lost;
        u64 qmd{}; ///< TEST: the QMD it could not read
    };

    /// Signals that wait, at most: a back that far behind leaves the view to its next takeover.
    static constexpr size_t MaxWaiting = 4096;

    /// A signal the front can't value, its table lost: no release of it is noted, and the view
    /// takes the back's table over once the back ran it.
    void Unvalued(const GpuSplit& split, Nx2AcquireStats::QmdLost lost) {
        checks_unvalued = true; // TEST
        Nx2AcquireStats::NoteQmdSignal(lost, WhatSent(split, lost));
    }

    /// TEST: what sent the last write the view could not read past, for `lost`.
    [[nodiscard]] static UnplacedKind WhatSent(const GpuSplit& split,
                                               Nx2AcquireStats::QmdLost lost) {
        return lost == Nx2AcquireStats::QmdLost::Mappings ? split.LastMappingKind()
                                                          : split.LastUnplacedKind();
    }

    /// TEST (EDEN_NX2_ACQUIRE_STATS): what wrote the QMD a signal sent at `position` waits for, as
    /// it was sent: a release into it the front noted, else the engine method whose entry ends at
    /// the write's position, else another entry (a CPU function, a deferred call). Logs the first.
    template <typename Memory>
    void NoteWaitWriter(const GpuSplit& split, const Memory& memory, u64 position,
                        const Outcome& outcome) const {
        const u64 target = *outcome.retry_after;
        std::string writer;
        if (const auto device = memory.Device(outcome.qmd)) {
            for (u64 offset = 0; offset < sizeof(Descriptor) && writer.empty(); offset += 4) {
                const auto queued = split.QueuedSemaphoreWriteAt(*device + offset);
                if (queued && queued->position == target) {
                    writer = fmt::format("a release into word {}", offset / 4);
                }
            }
        }
        if (writer.empty()) {
            writer = split.MethodEndingAt(target);
        }
        if (writer.empty()) {
            writer = "another entry";
        }
        static u32 logged{};
        if (logged < 16) {
            ++logged;
            LOG_INFO(HW_GPU,
                     "QMD view: a signal waits for a write into the QMD at {:#x} by {}, sent {} "
                     "ring words before it; the back is {} words behind",
                     outcome.qmd, writer, position - target, split.Backlog());
        }
        Nx2AcquireStats::NoteQmdWaitWriter(writer);
    }

    /// The signal sent at `position` waits, first (`why`: what it waits for) or behind the others.
    /// None of its releases is noted: they are placed once it is valued.
    void Wait(const GpuSplit& split, u64 position, const void* memory_id, GPUVAddr address,
              u32 argument, const Descriptor* inline_qmd,
              std::optional<Nx2AcquireStats::QmdLost> why) {
        checks_unvalued = true; // TEST
        if (waiting.size() >= MaxWaiting) {
            Nx2AcquireStats::NoteQmdLateLost(Nx2AcquireStats::QmdLost::TooMany,
                                             waiting.size()); // TEST
            synced = false;
            ClearWaiting();
            Nx2AcquireStats::NoteQmdSignal(Nx2AcquireStats::QmdLost::NotSynced); // TEST
            return;
        }
        Nx2AcquireStats::NoteQmdWait(why, why ? WhatSent(split, *why) : UnplacedKind::Other);
        WaitingSignal& signal = waiting.emplace_back();
        signal.position = position;
        signal.memory = memory_id;
        signal.address = address;
        signal.argument = argument;
        signal.has_inline = inline_qmd != nullptr;
        if (inline_qmd != nullptr) {
            signal.inline_qmd = *inline_qmd;
        }
        if (checks_pending) {
            // TEST: the back keeps what it reads for the signals of this entry from here on.
            signal.index = entry_signals;
            if (check_wait_position != position) {
                check_wait_position = position;
                check_first_waiting = entry_signals;
            }
        }
    }

    void ClearWaiting() {
        waiting.clear();
        late_reads.clear();
        retry_after = 0;
    }

    /// Values the waiting signals that read `memory` in order, from the first, as long as the back
    /// ran what the first waits for.
    template <typename Memory>
    void CatchUp(GpuSplit& split, const Memory& memory) {
        while (!waiting.empty()) {
            const WaitingSignal& first = waiting.front();
            if (first.memory != memory.Id() || !split.Ran(retry_after)) {
                return;
            }
            const Outcome outcome =
                Attempt(split, memory, first.position, first.address, first.argument,
                        first.has_inline ? &first.inline_qmd : nullptr, true);
            if (outcome.retry_after) {
                retry_after = *outcome.retry_after;
                return;
            }
            if (!outcome.valued) {
                Nx2AcquireStats::NoteQmdLateLost(outcome.lost, waiting.size()); // TEST
                synced = false;
                ClearWaiting();
                return;
            }
            if (Nx2DrainCheck::g_enabled) {
                // TEST: what it read, for the back to compare (CheckLateQmdReads).
                late_checks.push_back({first.position, first.index, std::move(late_reads_read)});
                late_reads_read.clear();
                table_check_due = table_check_due || waiting.size() == 1;
            }
            waiting.pop_front();
            retry_after = 0;
            Nx2AcquireStats::NoteQmdLate(waiting.empty()); // TEST
        }
        // Every signal sent before the reads made late is valued: none of their releases waits.
        late_reads.clear();
    }

    /// Runs the signal sent at `position` on the table as the back runs it, and notes what its
    /// launches release, or places it if `late`. Takes it back where the front can't read a QMD
    /// as the back reads it.
    template <typename Memory>
    Outcome Attempt(GpuSplit& split, const Memory& memory, u64 position, GPUVAddr address,
                    u32 argument, const Descriptor* inline_qmd, bool late) {
        using Lost = Nx2AcquireStats::QmdLost;
        // The back finds the mappings every change sent before the signal made, and none after it.
        const u64 mapping_change = split.LastMappingChange();
        if (mapping_change > position) {
            return {false, std::nullopt, Lost::MappedAfter};
        }
        if (!split.Ran(mapping_change)) {
            return {false, mapping_change, Lost::Mappings};
        }
        releases.clear();
        attempt_reads.clear();
        late_reads_read.clear();                         // TEST
        const size_t checked_reads = check_reads.size(); // TEST
        std::optional<Outcome> failure;
        const auto read = [&](u64 qmd_address, Descriptor& descriptor) {
            if (!memory.IsMapped(qmd_address, sizeof(descriptor))) {
                return false;
            }
            if (inline_qmd != nullptr && qmd_address == address) {
                descriptor = *inline_qmd;
            } else if (const auto unread =
                           ReadDescriptor(split, memory, qmd_address, position, descriptor)) {
                failure = unread;
                return false; // Stops the scheduling, which is taken back
            }
            // A release this signal ran before writes into the QMD.
            for (const Release& earlier : releases) {
                if (earlier.address + 16 > qmd_address &&
                    earlier.address < qmd_address + sizeof(descriptor)) {
                    failure = Outcome{false, std::nullopt, Lost::ReleasedInto};
                    return false;
                }
            }
            attempt_reads.push_back(qmd_address);
            if (Nx2DrainCheck::g_enabled) {
                // TEST: what the back reads too.
                (late ? late_reads_read : check_reads).push_back({qmd_address, descriptor});
            }
            return true;
        };
        const auto release = [&](u64 destination, u32 payload, u32 flags) {
            const bool four_words = (flags >> 30) == 0;
            if (!memory.IsMapped(destination, four_words ? 16 : sizeof(u32))) {
                return false;
            }
            releases.push_back({destination, payload, flags});
            return true;
        };
        scheduler.BeginAttempt();
        scheduler.Signal(address, argument, read, release, [](const Descriptor&) { return true; });
        if (!failure && late && ReleasesIntoLateReads()) {
            failure = Outcome{false, std::nullopt, Lost::LateReleasedInto};
        }
        if (failure) {
            scheduler.RollbackAttempt();
            check_reads.resize(checked_reads); // TEST: tried again later
            return *failure;
        }
        scheduler.CommitAttempt();
        if (late) {
            for (const u64 read_address : attempt_reads) {
                late_reads.insert(read_address >> 8);
            }
            for (const Release& done : releases) {
                memory.NoteWriteAt(split, done.address, (done.flags >> 30) == 0 ? 16 : 4, position);
            }
        } else {
            for (const Release& noted : releases) {
                NoteRelease(split, memory, noted.address, noted.payload, noted.flags);
            }
        }
        return {true, std::nullopt, Lost::NotSynced};
    }

    /// Late: whether a release of the signal writes a QMD it read, or one a signal valued late
    /// before read. The back may have run the release before the front read that QMD.
    [[nodiscard]] bool ReleasesIntoLateReads() const {
        for (const Release& done : releases) {
            const u64 first = done.address >> 8;
            const u64 last = (done.address + 15) >> 8;
            for (const u64 read_address : attempt_reads) {
                if ((read_address >> 8) == first || (read_address >> 8) == last) {
                    return true;
                }
            }
            if (late_reads.contains(first) || late_reads.contains(last)) {
                return true;
            }
        }
        return false;
    }

    /// Reads into `descriptor` the QMD at `qmd_address` as the back read it, or will, running the
    /// signal sent at `position`, or tells why the front can't.
    template <typename Memory>
    [[nodiscard]] std::optional<Outcome> ReadDescriptor(const GpuSplit& split, const Memory& memory,
                                                        GPUVAddr qmd_address, u64 position,
                                                        Descriptor& descriptor) const {
        using Lost = Nx2AcquireStats::QmdLost;
        const auto device = memory.Device(qmd_address);
        if (!device || memory.GpuWritten(*device, sizeof(descriptor))) {
            return Outcome{false, std::nullopt, Lost::GpuWritten};
        }
        // The last write sent there, placed or not (the inline write covers the whole QMD).
        const u64 target = memory.WritesTarget(split, qmd_address, sizeof(descriptor));
        const Lost lost = target == split.LastUnplacedWrite() ? Lost::Unplaced : Lost::Pending;
        if (target > position) {
            // Sent after the signal (it waited): memory may hold it, the back read what was before.
            return Outcome{false, std::nullopt, Lost::WrittenAfter, qmd_address};
        }
        if (split.Ran(target)) {
            // Read only once Ran saw the back past the write: the back stores its position after
            // the writes before it. A read before that look may be older than the write. The back
            // can run a write published shortly before the signal (to an idle back, after a few
            // words) while the front reads, and store its position before the front looks.
            memory.Read(qmd_address, descriptor.data(), sizeof(descriptor));
            return std::nullopt;
        }
        if (const InlineWrite* written = QueuedInlineWrite(split, *device);
            written != nullptr && written->position == target) {
            descriptor = written->qmd;
            Nx2AcquireStats::NoteQmdInlineRead(); // TEST
            return std::nullopt;
        }
        return Outcome{false, target, lost, qmd_address};
    }

    /// The last inline QMD sent to the QMD at `device` that the back has not written yet.
    [[nodiscard]] const InlineWrite* QueuedInlineWrite(const GpuSplit& split, u64 device) const {
        const size_t count = (std::min)(inline_writes_count, inline_writes.size());
        for (size_t i = 1; i <= count; ++i) {
            const InlineWrite& written =
                inline_writes[(inline_writes_count - i) % inline_writes.size()];
            if (split.Ran(written.position)) {
                break; // The back wrote it, and the older ones
            }
            if (written.device == device) {
                return written.overwritten ? nullptr : &written;
            }
        }
        return nullptr;
    }

    /// A release the back runs with the entry just sent, as ProcessNativeMethod's: what the word
    /// holds once the back ran it.
    template <typename Memory>
    void NoteRelease(GpuSplit& split, const Memory& memory, GPUVAddr address, u32 payload,
                     u32 flags) {
        const bool four_words = (flags >> 30) == 0;
        const bool reduction = (flags & (1U << 26)) != 0;
        const bool maximum = !reduction && !four_words && (flags & (1U << 19)) != 0;
        const u64 size = four_words ? 16 : sizeof(u32);
        const auto device = memory.Device(address);
        // The word a reduction or a maximum combines the payload with, before this write.
        std::optional<u32> base;
        if (device && (reduction || maximum)) {
            base = WordAt(split, memory, address, *device);
        }
        memory.NoteWrite(split, address, size);
        if (!device) {
            return;
        }
        // An inline QMD this entry wrote before is no longer what memory holds there.
        for (size_t i = 1; i <= (std::min)(inline_writes_count, inline_writes.size()); ++i) {
            InlineWrite& written = inline_writes[(inline_writes_count - i) % inline_writes.size()];
            if (written.position != split.Position()) {
                break;
            }
            if (written.device < *device + size && *device < written.device + sizeof(Descriptor)) {
                written.overwritten = true;
            }
        }
        // TEST: what was noted (EDEN_NX2_DRAIN_CHECK, EDEN_NX2_ACQUIRE_STATS).
        const auto check = [this](u64 word, u32 value, bool known, bool at_least) {
            using Note = Nx2AcquireStats::QmdNote;
            Nx2AcquireStats::NoteQmdNote(!known     ? Note::Unknown
                                         : at_least ? Note::AtLeast
                                                    : Note::Exact);
            if (checks_pending) {
                checks.push_back({word, value, known, at_least});
            }
        };
        Nx2AcquireStats::NoteRelease(*device, Nx2AcquireStats::Releaser::Qmd);
        if (reduction) {
            const bool is_signed = ((flags >> 24) & 1) != 0;
            const u32 value =
                base ? Tegra::ReduceSemaphore((flags >> 20) & 7, is_signed, *base, payload) : 0;
            split.NoteSemaphoreWrite(*device, value, base.has_value());
            check(*device, value, base.has_value(), false);
        } else if (four_words) {
            // The payload as 64 bits, a timestamp after it. Fence-flagged reports, like the plain
            // release below: they land once their fence ran.
            split.NoteSemaphoreWrite(*device, payload, true, false, GpuSplit::Landing::Fenced);
            split.NoteSemaphoreWrite(*device + 4, 0, true, false, GpuSplit::Landing::Fenced);
            check(*device, payload, true, false);
            check(*device + 4, 0, true, false);
        } else if (maximum) {
            // Whatever the word held, it holds the payload at least.
            const u32 value = base ? (std::max)(*base, payload) : payload;
            split.NoteSemaphoreWrite(*device, value, true, !base.has_value());
            check(*device, value, true, !base.has_value());
        } else {
            split.NoteSemaphoreWrite(*device, payload, true, false, GpuSplit::Landing::Fenced);
            check(*device, payload, true, false);
        }
    }

    /// The word the back finds at `address` (`device`) when it runs the entry just sent, where
    /// the front knows it: the last release sent there, if nothing sent since may have written
    /// over it, else memory, once the back ran every write sent there before the entry. The entry
    /// itself is held unpublished (PublishHold): an idle back that ran it before this read would
    /// leave its own increment in memory, and the front would count it twice.
    template <typename Memory>
    [[nodiscard]] std::optional<u32> WordAt(const GpuSplit& split, const Memory& memory,
                                            GPUVAddr address, u64 device) const {
        // The back reads the syncpoint shim from the syncpoints.
        if (memory.InSyncpointShim(address, sizeof(u32))) {
            return std::nullopt;
        }
        // The last write sent to the word's granule, placed or not.
        const u64 target = memory.WritesTarget(split, address, sizeof(u32));
        if (const auto queued = split.QueuedSemaphoreWriteAt(device)) {
            // A fenced one lands once its fence ran, before or after the back reads the word.
            if (queued->known && !queued->at_least && !queued->fenced &&
                queued->position >= target) {
                return queued->payload;
            }
            return std::nullopt;
        }
        if (memory.GpuWritten(device, sizeof(u32)) || !split.Ran(target)) {
            return std::nullopt;
        }
        return memory.ReadWord(address);
    }

    Tegra::NativeQmdScheduler scheduler{Tegra::NativeQmdScheduler::OwnTable{}};
    /// The table the engines of every channel share, which the back's scheduling uses.
    Tegra::NativeQmdScheduler back_table;
    std::vector<Release> releases;
    std::vector<u64> attempt_reads; ///< The QMDs the attempt read
    bool synced{};
    /// Signals that wait to be valued, oldest first; the first may be once the back ran
    /// retry_after. late_reads: the granules of the QMDs read late since the waiting began.
    std::deque<WaitingSignal> waiting;
    u64 retry_after{};
    std::unordered_set<u64> late_reads;
    /// TEST (EDEN_NX2_DRAIN_CHECK): the entry the last signal came with and its signals before
    /// it; the entry whose signals wait and the first that waits there, for TakeChecks; what the
    /// last late attempt read; the signals valued late since TakeChecks; a table copy is due.
    u64 entry_position{};
    u32 entry_signals{};
    u64 check_wait_position{};
    u32 check_first_waiting{};
    std::vector<Nx2DrainCheck::QmdRead> late_reads_read;
    std::vector<Nx2DrainCheck::LateQmdRead> late_checks;
    bool table_check_due{};
    /// The last inline QMDs sent, oldest first from inline_writes_count on: a back a millisecond
    /// behind has some ten to write, a few times that in long frames.
    std::array<InlineWrite, 256> inline_writes{};
    size_t inline_writes_count{};
    /// TEST (EDEN_NX2_DRAIN_CHECK): see TakeChecks.
    std::vector<Nx2DrainCheck::QmdRelease> checks;
    std::vector<Nx2DrainCheck::QmdRead> check_reads;
    u64 table_checks{};
    bool checks_pending{};
    bool checks_unvalued{};
};

} // namespace VideoCommon
