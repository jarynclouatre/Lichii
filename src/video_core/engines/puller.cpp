// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <cstdlib>

#include "common/assert.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/nx2_profile.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/engines/puller.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {

namespace {

constexpr u32 Gf100BindClassMask = 0xffff;
constexpr u32 Gf100BindValidMask =  0x1f0000 | Gf100BindClassMask;

// Reads a semaphore word, flushing a GPU-written copy out of the buffer cache first.
u32 ReadSemaphoreWord(DmaPusher& dma_pusher, GPUVAddr address) {
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
        device && dma_pusher.rasterizer->MustFlushRegion(*device, sizeof(u32),
                                                          VideoCommon::CacheType::BufferCache)) {
        VideoCommon::GpuSplit::Get().Run(
            [rasterizer = dma_pusher.rasterizer, device = *device] {
                rasterizer->FlushRegion(device, sizeof(u32), VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::SemaphoreFlush);
    }
    return dma_pusher.memory_manager.Read<u32>(address);
}

// With the GPU thread split (gpu_split.h) the rasterizer belongs to the back thread: work that
// only feeds it runs there in order; reads of what it wrote first wait for it (DrainBack).
// `writes`: what it writes of guest memory, for the front's reads (GpuSplit::Writes).
template <typename Func>
void OnBack(const Func& func,
            VideoCommon::GpuSplit::Writes writes = VideoCommon::GpuSplit::Writes::Anywhere) {
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer(func, writes);
    } else {
        func();
    }
}

// Fence work (ReleaseFences, fences and syncpoints signaled): downloads land in guest memory
// where the GPU wrote, nothing else.
constexpr auto FenceWrites = VideoCommon::GpuSplit::Writes::Caches;

void DrainBack() {
    VideoCommon::GpuSplit::Get().Drain(VideoCommon::DrainSite::Acquire);
}

// EDEN_NX2_DRAIN_CHECK: whether an acquire (or a suspended one's poll) met its value before the
// drain as well as after it, and whether the GPU had released a semaphore at the address.
void NoteAcquireCheck(DmaPusher& dma_pusher, GPUVAddr address, bool poll, bool met_before,
                      bool met_after) {
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    const bool released = device && VideoCommon::Nx2DrainCheck::WasReleased(*device);
    VideoCommon::Nx2DrainCheck::NoteAcquire(poll, released, met_before, met_after);
}

// The fence manager records into the scheduler and downloads the CPU's writes over GPU data:
// the back's state, which a drained back still uses from its idle hook (it submits).
void ReleaseFences(DmaPusher& dma_pusher) {
    VideoCommon::GpuSplit::Get().Run(
        [rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
        VideoCommon::DrainSite::FenceRelease);
}

// A release with `properties` was just sent to the back (as Writes::Noted): tell the front what
// it writes (CheckWithoutDrain).
void NoteRelease(DmaPusher& dma_pusher, GPUVAddr address, u32 payload,
                 VideoCommon::QueryPropertiesFlags properties) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active()) {
        return;
    }
    const bool four_words = True(properties & VideoCommon::QueryPropertiesFlags::HasTimeout);
    // A fence-flagged report lands once the fence after it ran (QueryCacheBase::CounterReport).
    const auto landing = True(properties & VideoCommon::QueryPropertiesFlags::IsAFence)
                             ? VideoCommon::GpuSplit::Landing::Fenced
                             : VideoCommon::GpuSplit::Landing::AtRun;
    split.NoteWrite(dma_pusher.memory_manager, address, four_words ? 16 : sizeof(u32));
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device) {
        return;
    }
    split.NoteSemaphoreWrite(*device, payload, true, false, landing);
    VideoCommon::Nx2AcquireStats::NoteRelease(*device,
                                              VideoCommon::Nx2AcquireStats::Releaser::Puller);
    if (four_words) {
        // The payload as 64 bits (a timestamp follows).
        split.NoteSemaphoreWrite(*device + 4, 0, true, false, landing);
    }
}

// Whether the back writes a payload report without a fence flag to memory when it runs it
// (CounterReport): below High GPU accuracy, with fences that run their operations at once
// (FenceManager::SignalFence). Fence-flagged reports land once their fence ran
// (GpuSplit::Landing::Fenced).
bool ReleasesLandAtOnce() {
    const bool delay_fence =
        Settings::IsGPUFenceBehaviorDefault()
            ? Settings::IsGPULevelHigh()
            : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
    return !delay_fence && !Settings::IsGPULevelHigh();
}

// Whether the front decides Switch 2 acquires from its view of the semaphore words (AskView)
// instead of draining the split: the back writes each release to memory as it runs it, or once
// its fence ran (GpuSplit::Landing), and the front sees every release it sends, its macros'
// included.
bool ViewApplies() {
    const auto& split = VideoCommon::GpuSplit::Get();
    return split.Active() && split.FrontMacros() && ReleasesLandAtOnce();
}

// What the front can tell of an acquire without waiting for the back.
struct ViewAnswer {
    enum class Kind {
        Met,         ///< By the word in memory
        MetByQueued, ///< By the last release sent to the word, which the back has not run yet
        NotMet,      ///< Nothing sent to the word meets it yet
        Unknown,     ///< Memory tells once the back ran the work before `position`
        Flush,       ///< Shaders wrote the word: the buffer cache may hold it newer than memory
    };
    Kind kind;
    u64 position{};
};

ViewAnswer AskView(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire, u64 device) {
    using Kind = ViewAnswer::Kind;
    const auto& split = VideoCommon::GpuSplit::Get();
    const auto memory = [&] { return dma_pusher.memory_manager.Read<u32>(acquire.address); };
    // The words Switch 2 syncs on only advance (NVN2 even releases QMD completions as maxima,
    // KeplerCompute::ProcessNativeMethod): memory that meets a greater-or-equal acquire now still
    // does once the back ran the queued work, also where the buffer cache holds it newer.
    if (Nx2MayBeGpuWritten(device, sizeof(u32)) &&
        dma_pusher.rasterizer->MustFlushRegion(device, sizeof(u32),
                                               VideoCommon::CacheType::BufferCache)) {
        return {acquire.WaitsToReach() && acquire.Matches(memory()) ? Kind::Met : Kind::Flush};
    }
    // Another acquire holds only for the value the word has when the channel reaches it. Compute
    // and copy work sent before it may still release the word without the front's knowing (QMD
    // releases): a value that meets it stands once the back ran that work.
    const bool settled = acquire.WaitsToReach() || split.Ran(acquire.writers_before);
    // The last release the front sent to the word decides while it may not be in memory.
    const auto queued = split.QueuedSemaphoreWriteAt(device);
    if (queued && queued->fenced) {
        // It lands once a fence ran, at no position the front can wait for: memory decides, as
        // after a drain, and the landing latches the waiter or a poll finds the word. Writes sent
        // to the word before it are in memory once the back ran it. A greater-or-equal acquire
        // memory meets now meets it then too (the words only advance).
        if (!split.Ran(queued->position) && !(queued->known && acquire.WaitsToReach())) {
            return {Kind::Unknown, queued->position};
        }
    } else if (queued) {
        if (!queued->known) {
            return {Kind::Unknown, queued->position};
        }
        if (queued->at_least) {
            // A maximum (QMD releases): the word holds the payload or more once the back ran it,
            // which meets a greater-or-equal acquire of the payload or less. Else memory tells.
            return acquire.WaitsToReach() && acquire.Matches(queued->payload)
                       ? ViewAnswer{Kind::MetByQueued}
                       : ViewAnswer{Kind::Unknown, queued->position};
        }
        if (!acquire.Matches(queued->payload)) {
            return {Kind::NotMet};
        }
        if (settled || queued->position >= acquire.writers_before) {
            return {Kind::MetByQueued};
        }
        return {Kind::Unknown, acquire.writers_before};
    }
    if (!acquire.Matches(memory())) {
        return {Kind::NotMet};
    }
    return settled ? ViewAnswer{Kind::Met} : ViewAnswer{Kind::Unknown, acquire.writers_before};
}

// Has the back flush the word out of the buffer cache after the work sent before: memory holds
// it once the back ran the flush (PendingAcquire::flush_position).
void RequestFlush(DmaPusher& dma_pusher, Puller::PendingAcquire& acquire, u64 device) {
    auto& split = VideoCommon::GpuSplit::Get();
    split.Defer(
        [rasterizer = dma_pusher.rasterizer, device] {
            rasterizer->FlushRegion(device, sizeof(u32), VideoCommon::CacheType::BufferCache);
        },
        VideoCommon::GpuSplit::Writes::Caches);
    acquire.flush_position = split.Position();
    acquire.flushed = true;
}

// EDEN_NX2_DRAIN_CHECK: what the drain path decides instead, to verify the view (a release the
// back ran latches `waiter`, a suspended acquire's).
bool MetAfterDrain(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire,
                   const void* waiter) {
    DrainBack();
    return acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address)) ||
           (waiter != nullptr && Nx2SemaphoreWaiters::Get().Latched(waiter));
}

// TEST (EDEN_NX2_DRAIN_CHECK): whether a fence-flagged release to the word at `device` may still
// land. One that lands during the drain can change the word the view read, as it can on the
// drain path itself.
bool FencedWritePending(u64 device) {
    const auto queued = VideoCommon::GpuSplit::Get().QueuedSemaphoreWriteAt(device);
    return queued && queued->fenced;
}

// TEST: checks a pass of the view against the drain path: whether the drained word meets it, and
// whether a fenced release that landed during the drain changed it.
bool PassStands(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire, const void* waiter,
                u64 device, bool poll, VideoCommon::Nx2DrainCheck::ViewPass how) {
    const bool fenced = FencedWritePending(device);
    const bool met = MetAfterDrain(dma_pusher, acquire, waiter);
    VideoCommon::Nx2DrainCheck::NoteViewPass(poll, how, met,
                                             !met && fenced && !FencedWritePending(device));
    return met;
}

// TEST: counts the view's decision on an acquire (EDEN_NX2_ACQUIRE_STATS) and checks it against
// the drain path (EDEN_NX2_DRAIN_CHECK): a pass the drained word doesn't meet suspends instead,
// a suspension the drained word meets passes.
ViewAnswer CheckAcquire(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire,
                        ViewAnswer answer, bool waited, u64 device) {
    using Kind = ViewAnswer::Kind;
    namespace Check = VideoCommon::Nx2DrainCheck;
    namespace Stats = VideoCommon::Nx2AcquireStats;
    const bool pass = answer.kind == Kind::Met || answer.kind == Kind::MetByQueued;
    const auto how = waited                             ? Check::ViewPass::Waited
                     : answer.kind == Kind::MetByQueued ? Check::ViewPass::Queued
                                                        : Check::ViewPass::Memory;
    if (Stats::g_enabled) {
        const auto decision = !pass                            ? Stats::Decision::Suspended
                              : how == Check::ViewPass::Waited ? Stats::Decision::Waited
                              : how == Check::ViewPass::Queued ? Stats::Decision::Queued
                                                               : Stats::Decision::Memory;
        Stats::NoteAcquire(acquire.operation, decision, device, waited,
                           Nx2MayBeGpuWritten(device, sizeof(u32)), answer.kind == Kind::Flush);
    }
    if (Check::g_enabled) {
        if (pass) {
            if (!PassStands(dma_pusher, acquire, nullptr, device, false, how)) {
                return {Kind::NotMet};
            }
        } else {
            const bool met = MetAfterDrain(dma_pusher, acquire, nullptr);
            Check::NoteViewWait(false, met);
            if (met) {
                return {Kind::Met};
            }
        }
    }
    return answer;
}

// TEST: whether the view's resume of a poll stands (EDEN_NX2_DRAIN_CHECK).
bool PollPassStands(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire,
                    const void* waiter, u64 device, VideoCommon::Nx2DrainCheck::ViewPass how) {
    if (!VideoCommon::Nx2DrainCheck::g_enabled) {
        return true;
    }
    return PassStands(dma_pusher, acquire, waiter, device, true, how);
}

// TEST: a suspended acquire resumes (EDEN_NX2_ACQUIRE_STATS).
void NoteResumed(const void* waiter, const Puller::PendingAcquire& acquire,
                 VideoCommon::Nx2AcquireStats::Resume how) {
    namespace Stats = VideoCommon::Nx2AcquireStats;
    if (!Stats::g_enabled) {
        return;
    }
    const auto source = how == Stats::Resume::Latch ? Nx2SemaphoreWaiters::Get().LatchSource(waiter)
                                                    : Stats::Source::Unknown;
    Stats::NoteResume(how, source, acquire.flushed,
                      std::chrono::steady_clock::now() - acquire.since);
}

// The fence work an acquire that still waits after the drain sends (ReleaseFences), sent once the
// back ran everything before the acquire: the release that work runs then counts as it did (the
// waiter is registered), and acquires the queued work releases send none.
void MaybeReleaseFences(DmaPusher& dma_pusher, Puller::PendingAcquire& acquire) {
    if (acquire.fences_released || !VideoCommon::GpuSplit::Get().Ran(acquire.sent_at)) {
        return;
    }
    acquire.fences_released = true;
    OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); }, FenceWrites);
}

// Logs once an acquire that waits for more than 2 s. `current`: the word as read after a drain,
// or nothing to read it from memory.
void ReportLongWait(DmaPusher& dma_pusher, Puller::PendingAcquire& acquire,
                    std::optional<u32> current) {
    if (acquire.reported ||
        std::chrono::steady_clock::now() - acquire.since <= std::chrono::seconds{2}) {
        return;
    }
    acquire.reported = true;
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    const GPUVAddr words = acquire.address & ~GPUVAddr{0xf};
    LOG_WARNING(
        HW_GPU,
        "Channel {} waits on semaphore {:#x} (op {}, {:#x}, now {:#x}) for 2 s; device "
        "{:#x}, buffer cache holds GPU data: {}, words {:08x} {:08x} {:08x} {:08x}",
        dma_pusher.channel_state.bind_id, acquire.address, acquire.operation, acquire.value,
        current.value_or(dma_pusher.memory_manager.Read<u32>(acquire.address)), device.value_or(0),
        device && dma_pusher.rasterizer->MustFlushRegion(*device, 16, VideoCommon::CacheType::All),
        dma_pusher.memory_manager.Read<u32>(words), dma_pusher.memory_manager.Read<u32>(words + 4),
        dma_pusher.memory_manager.Read<u32>(words + 8),
        dma_pusher.memory_manager.Read<u32>(words + 12));
}

} // Anonymous namespace

bool Puller::PendingAcquire::Matches(u32 current) const {
    switch (static_cast<GpuSemaphoreOperation>(operation)) {
    case GpuSemaphoreOperation::AcquireEqual:
        return current == value;
    case GpuSemaphoreOperation::AcquireGequal:
        return current >= value;
    case GpuSemaphoreOperation::AcquireMask:
        return (current & value) != 0;
    default:
        return true;
    }
}

bool Puller::PendingAcquire::WaitsToReach() const {
    return static_cast<GpuSemaphoreOperation>(operation) == GpuSemaphoreOperation::AcquireGequal;
}

bool Puller::CanResumeAcquire(DmaPusher& dma_pusher) {
    if (!pending_acquire || pending_acquire->met) {
        return true;
    }
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(pending_acquire->address);
        device && ViewApplies()) {
        pending_acquire->met = PollWithoutDrain(dma_pusher, *device);
        return pending_acquire->met;
    }
    const bool check = VideoCommon::Nx2DrainCheck::g_enabled;
    const bool met_before =
        check &&
        pending_acquire->Matches(dma_pusher.memory_manager.Read<u32>(pending_acquire->address));
    // The release may still be queued for the split GPU thread's back half (and the caches
    // ReadSemaphoreWord asks are its state).
    DrainBack();
    const u32 current = ReadSemaphoreWord(dma_pusher, pending_acquire->address);
    if (check) [[unlikely]] {
        NoteAcquireCheck(dma_pusher, pending_acquire->address, true, met_before,
                         pending_acquire->Matches(current));
    }
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        if (static std::atomic<u32> rescued{};
            !pending_acquire->Matches(current) && rescued.fetch_add(1) < 64) {
            LOG_INFO(HW_GPU, "Channel {} met semaphore {:#x} value {:#x} at its release (now {:#x})",
                     dma_pusher.channel_state.bind_id, pending_acquire->address,
                     pending_acquire->value, current);
        }
        return true;
    }
    if (pending_acquire->Matches(current)) {
        return true;
    }
    // With delayed fence operations (Balanced or Accurate fences, accurate GPU), a release
    // queued after the acquire released the fences waits in the caches for the next fence. With
    // every channel waiting none comes: release them again now and then while waiting. (The
    // semaphore view, PollWithoutDrain, applies only while releases without a fence flag land at
    // once: never with delayed fence operations. Fence-flagged ones signal their own fence.)
    const bool delayed = Settings::IsGPUFenceBehaviorDefault()
                             ? Settings::IsGPULevelHigh()
                             : Settings::IsGPUFenceBehaviorBalanced() ||
                                   Settings::IsGPUFenceBehaviorAccurate();
    if (const auto now = std::chrono::steady_clock::now();
        delayed && now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
        pending_acquire->last_fences = now;
        ReleaseFences(dma_pusher);
        if (Nx2SemaphoreWaiters::Get().Latched(this) ||
            pending_acquire->Matches(ReadSemaphoreWord(dma_pusher, pending_acquire->address))) {
            return true;
        }
    }
    ReportLongWait(dma_pusher, *pending_acquire, current);
    return false;
}

bool Puller::PollWithoutDrain(DmaPusher& dma_pusher, u64 device) {
    using Kind = ViewAnswer::Kind;
    using Resume = VideoCommon::Nx2AcquireStats::Resume;
    using ViewPass = VideoCommon::Nx2DrainCheck::ViewPass;
    PendingAcquire& acquire = *pending_acquire;
    auto& split = VideoCommon::GpuSplit::Get();
    // A release the back ran latches the waiter, even if a later one changed the word again.
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        NoteResumed(this, acquire, Resume::Latch);
        return true;
    }
    if (acquire.flush_position && split.Ran(*acquire.flush_position)) {
        acquire.flush_position.reset();
    }
    // TEST: a resume the drained word contradicts was checked already.
    bool checked = false;
    if (!acquire.flush_position) {
        const ViewAnswer answer = AskView(dma_pusher, acquire, device);
        switch (answer.kind) {
        case Kind::Met:
            if (!PollPassStands(dma_pusher, acquire, this, device, ViewPass::Memory)) {
                checked = true;
                break;
            }
            NoteResumed(this, acquire, Resume::Memory);
            return true;
        case Kind::MetByQueued:
            if (!PollPassStands(dma_pusher, acquire, this, device, ViewPass::Queued)) {
                checked = true;
                break;
            }
            // What the front reads after the acquire waits for the work sent before the release.
            split.NoteSync();
            NoteResumed(this, acquire, Resume::Queued);
            return true;
        case Kind::Flush:
            RequestFlush(dma_pusher, acquire, device);
            break;
        case Kind::NotMet:
        case Kind::Unknown:
            // Poll again later, without waiting for the back: until it runs a release that meets
            // the acquire (which latches the waiter) or the work before `position`, nothing the
            // front sees meets it.
            break;
        }
    }
    if (VideoCommon::Nx2DrainCheck::g_enabled && !checked) [[unlikely]] {
        const bool met = MetAfterDrain(dma_pusher, acquire, this);
        VideoCommon::Nx2DrainCheck::NoteViewWait(true, met);
        if (met) {
            NoteResumed(this, acquire, Resume::Drain);
            return true;
        }
    }
    MaybeReleaseFences(dma_pusher, acquire);
    ReportLongWait(dma_pusher, acquire, std::nullopt);
    return false;
}

bool Puller::ResumeAcquire(DmaPusher& dma_pusher) {
    if (!CanResumeAcquire(dma_pusher)) {
        return false;
    }
    // Only a resume from a suspension counts: DmaPusher::Step calls this before every entry, and
    // counting those made the count odd ("suspended") at random, so submit-time snapshots were
    // thrown away for about half of all entries (CommandList::snapshot_suspensions).
    if (pending_acquire) {
        pending_acquire.reset();
        dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
        Nx2SemaphoreWaiters::Get().Remove(this);
    }
    return true;
}

void Puller::ProcessBindMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_DEBUG(HW_GPU, "Binding subchannel {} to engine {:#x}", method_call.subchannel, method_call.argument);
    u32 engine = method_call.argument;
    if ((engine & ~Gf100BindClassMask) != 0 && (engine & ~Gf100BindValidMask) == 0) {
        engine &= Gf100BindClassMask;
    }
    // Switch 2 binds Ampere classes; run them on the Maxwell engines (same method layout for
    // most state). Class 0 unbinds the subchannel.
    switch (engine & 0xFFFF) {
    case 0xC797: // AMPERE_B
        engine = static_cast<u32>(EngineID::MAXWELL_B);
        break;
    case 0xC7C0: // AMPERE_COMPUTE_B
        engine = static_cast<u32>(EngineID::KEPLER_COMPUTE_B);
        break;
    case 0xC7B5: // AMPERE_DMA_COPY_B
        engine = static_cast<u32>(EngineID::MAXWELL_DMA_COPY_A);
        break;
    case 0x0:
        if (g_nx2_ampere) {
            LOG_DEBUG(HW_GPU, "Switch 2 unbinds subchannel {}", method_call.subchannel);
            return;
        }
        break;
    default:
        break;
    }

    const auto engine_id = static_cast<EngineID>(engine);
    bound_engines[method_call.subchannel] = engine_id;
    switch (engine_id) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->fermi_2d, method_call.subchannel, EngineTypes::Fermi2D);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_3d, method_call.subchannel, EngineTypes::Maxwell3D);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_compute, method_call.subchannel, EngineTypes::KeplerCompute);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_dma, method_call.subchannel, EngineTypes::MaxwellDMA);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_memory, method_call.subchannel, EngineTypes::KeplerMemory);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->nv01_timer, method_call.subchannel, EngineTypes::Nv01Timer);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine {:04X}", engine_id);
        break;
    }
}

void Puller::ProcessFenceActionMethod(DmaPusher& dma_pusher) {
    switch (regs.fence_action.op) {
    case Puller::FenceOperation::Acquire:
        // UNIMPLEMENTED_MSG("Channel Scheduling pending.");
        // WaitFence(regs.fence_action.syncpoint_id, regs.fence_value);
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); }, FenceWrites);
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case Puller::FenceOperation::Increment:
        OnBack([rasterizer = dma_pusher.rasterizer,
                id = static_cast<u32>(
                    regs.fence_action.syncpoint_id.Value())] { rasterizer->SignalSyncPoint(id); },
               FenceWrites);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented operation {}", regs.fence_action.op.Value());
        break;
    }
}

void Puller::ProcessSemaphoreTriggerMethod(DmaPusher& dma_pusher) {
    const auto semaphoreOperationMask = 0xF;
    const auto op = GpuSemaphoreOperation(regs.semaphore_trigger & semaphoreOperationMask);
    // EDEN_NX2_SEM_WATCH=<gpu address>: log every release/acquire on the 16 bytes there.
    static const GPUVAddr watch = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
        return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
    }();
    if (watch != 0 && (regs.semaphore_address.SemaphoreAddress() & ~GPUVAddr{0xf}) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} as {} device {:#x} op {} {:#x} value {:#x} (now {:#x})",
                 dma_pusher.channel_state.bind_id, dma_pusher.memory_manager.GetID(),
                 dma_pusher.memory_manager.GpuToCpuAddress(regs.semaphore_address.SemaphoreAddress()).value_or(0),
                 static_cast<u32>(op),
                 regs.semaphore_address.SemaphoreAddress(), regs.semaphore_sequence,
                 dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress()));
    }
    if (op == GpuSemaphoreOperation::WriteLong) {
        const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
        const u32 payload = regs.semaphore_sequence;
        // RELEASE_SIZE (bit 24) set means a 4-byte release without the timestamp; writing 16
        // bytes there clobbers the words next to it (Switch 2 packs semaphores tightly).
        const auto properties = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0
                                    ? VideoCommon::QueryPropertiesFlags::IsAFence
                                    : VideoCommon::QueryPropertiesFlags::HasTimeout;
        OnBack(
            [rasterizer = dma_pusher.rasterizer, sequence_address, properties, payload] {
                rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload, properties,
                                  payload, 0);
            },
            VideoCommon::GpuSplit::Writes::Noted);
        NoteRelease(dma_pusher, sequence_address, payload, properties);
    } else if (g_nx2_ampere) {
        // Switch 2 waits on semaphores the CPU (or another channel) releases, e.g. once it has
        // written the commands this channel runs next: suspend the channel until the value is
        // met instead of reading ahead (approach from changes.patch).
        PendingAcquire acquire{
            .address = regs.semaphore_address.SemaphoreAddress(),
            .value = regs.semaphore_sequence,
            .operation = static_cast<u32>(op),
            .since = std::chrono::steady_clock::now(),
            .last_fences = std::chrono::steady_clock::now(),
            .reported = false,
        };
        if (op != GpuSemaphoreOperation::AcquireEqual && op != GpuSemaphoreOperation::AcquireGequal &&
            op != GpuSemaphoreOperation::AcquireMask) {
            LOG_ERROR(HW_GPU, "Invalid semaphore operation {}", static_cast<u32>(op));
            return;
        }
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireChecks, 1);
        if (AcquireWithoutDrain(dma_pusher, acquire)) {
            return;
        }
        if (VideoCommon::Nx2AcquireStats::g_enabled) [[unlikely]] {
            VideoCommon::Nx2AcquireStats::NoteAcquire(
                acquire.operation, VideoCommon::Nx2AcquireStats::Decision::Drained, 0, false, false,
                false);
        }
        const bool check = VideoCommon::Nx2DrainCheck::g_enabled;
        const bool met_before =
            check && acquire.Matches(dma_pusher.memory_manager.Read<u32>(acquire.address));
        DrainBack();
        const bool met = acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address));
        if (check) [[unlikely]] {
            NoteAcquireCheck(dma_pusher, acquire.address, false, met_before, met);
        }
        if (met) {
            return;
        }
        // Registered before the fences run, so a release among them that meets the value counts
        // even if a later one overwrites it.
        auto& waiters = Nx2SemaphoreWaiters::Get();
        if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address)) {
            waiters.Add(this, *device, acquire.value, acquire.operation);
        }
        ReleaseFences(dma_pusher);
        if (!waiters.Latched(this) && !acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address))) {
            pending_acquire = acquire;
            dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
        } else {
            waiters.Remove(this);
        }
    } else {
        DrainBack();
        do {
            const u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
            regs.acquire_source = true;
            regs.acquire_value = regs.semaphore_sequence;
            if (op == GpuSemaphoreOperation::AcquireEqual) {
                regs.acquire_active = true;
                regs.acquire_mode = false;
                if (word != regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireGequal) {
                regs.acquire_active = true;
                regs.acquire_mode = true;
                if (word < regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireMask) {
                if (word && regs.semaphore_sequence == 0) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else {
                LOG_ERROR(HW_GPU, "Invalid semaphore operation");
            }
        } while (false);
    }
}

bool Puller::AcquireWithoutDrain(DmaPusher& dma_pusher, PendingAcquire& acquire) {
    using Kind = ViewAnswer::Kind;
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    if (!device || !ViewApplies()) {
        return false;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    acquire.writers_before = split.LastWriterPosition();
    acquire.sent_at = split.Position();
    ViewAnswer answer = AskView(dma_pusher, acquire, *device);
    // The value is in memory once the back ran that work, before what was sent after it. A
    // release the front can't value, then the writers before the acquire: two waits at most.
    u32 waits = 0;
    for (; answer.kind == Kind::Unknown && waits < 2; ++waits) {
        split.WaitForPosition(answer.position, VideoCommon::DrainSite::Acquire);
        answer = AskView(dma_pusher, acquire, *device);
    }
    if (VideoCommon::Nx2DrainCheck::g_enabled || VideoCommon::Nx2AcquireStats::g_enabled)
        [[unlikely]] {
        answer = CheckAcquire(dma_pusher, acquire, answer, waits != 0, *device);
    }
    switch (answer.kind) {
    case Kind::Met:
        return true;
    case Kind::MetByQueued:
        // What the front reads after the acquire waits for the work sent before the release
        // (GpuSplit::NoteSync).
        split.NoteSync();
        return true;
    case Kind::Flush:
        RequestFlush(dma_pusher, acquire, *device);
        break;
    case Kind::NotMet:
    case Kind::Unknown:
        break;
    }
    // Suspended: a release the back runs latches the waiter, registered before any fence work,
    // and the polls see the others in memory or in the view (PollWithoutDrain).
    Nx2SemaphoreWaiters::Get().Add(this, *device, acquire.value, acquire.operation);
    MaybeReleaseFences(dma_pusher, acquire);
    pending_acquire = acquire;
    dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
    return true;
}

void Puller::ProcessSemaphoreRelease(DmaPusher& dma_pusher) {
    const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
    const u32 payload = regs.semaphore_release;
    OnBack(
        [rasterizer = dma_pusher.rasterizer, sequence_address, payload] {
            rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload,
                              VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
        },
        VideoCommon::GpuSplit::Writes::Noted);
    NoteRelease(dma_pusher, sequence_address, payload,
                VideoCommon::QueryPropertiesFlags::IsAFence);
}

void Puller::ProcessSemaphoreAcquire(DmaPusher& dma_pusher) {
    DrainBack();
    // The CPU may rewrite what the next draws read once it sees the release this waits for.
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
    const auto value = regs.semaphore_acquire;
    while (word != value) {
        regs.acquire_active = true;
        regs.acquire_value = value;
        ReleaseFences(dma_pusher);
        word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
        // TODO(kemathe73) figure out how to do the acquire_timeout
        regs.acquire_mode = false;
        regs.acquire_source = false;
    }
}

// Ampere SEM_EXECUTE (method 0x1B): operation in bits 2:0 on the semaphore at
// SEM_ADDR_HI:SEM_ADDR_LO (methods 0x18/0x17) with payload SEM_PAYLOAD_LO (0x19).
void Puller::ProcessSemaphoreExecute(DmaPusher& dma_pusher, u32 argument) {
    const GPUVAddr address = (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                             (regs.reg_array[0x17] & ~3U);
    const u32 payload = regs.reg_array[0x19];
    const u32 operation = argument & 7;
    if (operation == 1) { // RELEASE
        OnBack(
            [rasterizer = dma_pusher.rasterizer, address, payload] {
                rasterizer->Query(address, VideoCommon::QueryType::Payload,
                                  VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
            },
            VideoCommon::GpuSplit::Writes::Noted);
        NoteRelease(dma_pusher, address, payload, VideoCommon::QueryPropertiesFlags::IsAFence);
        return;
    }
    // Acquires: never block the GPU thread on them; they are checked for the warning below only.
    const auto ready = [operation, payload](u32 value) {
        return operation == 0   ? value == payload
               : operation == 2 ? value >= payload
               : operation == 3 ? static_cast<s32>(value - payload) >= 0
               : operation == 4 ? (value & payload) != 0
               : operation == 5 ? (value & payload) == 0
                                : true;
    };
    std::optional<u32> word;
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
        device && ViewApplies()) {
        // The last release sent to the word tells while the back hasn't run it. What the front
        // reads after the acquire waits for the work sent before it, as after a drain.
        auto& split = VideoCommon::GpuSplit::Get();
        split.NoteSync();
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); }, FenceWrites);
        if (const auto queued = split.QueuedSemaphoreWriteAt(*device)) {
            if (queued->known && !queued->at_least) {
                word = queued->payload;
            }
        } else {
            word = dma_pusher.memory_manager.Read<u32>(address);
        }
    } else {
        DrainBack();
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
        ReleaseFences(dma_pusher);
        word = dma_pusher.memory_manager.Read<u32>(address);
    }
    if (word && !ready(*word)) {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1) < 8) {
            LOG_WARNING(HW_GPU, "SEM_EXECUTE op {} at {:#x}: {:#x} vs {:#x} not met, continuing",
                        operation, address, *word, payload);
        }
    }
}

/// Calls a GPU puller method.
void Puller::CallPullerMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    static const bool nx2_trace = std::getenv("EDEN_NX2_PULLER_TRACE") != nullptr;
    if (nx2_trace) {
        LOG_INFO(HW_GPU, "nx2 puller sub={} method={:#x} arg={:#x}", method_call.subchannel,
                 method_call.method, method_call.argument);
    }
    regs.reg_array[method_call.method] = method_call.argument;
    const auto method = static_cast<BufferMethods>(method_call.method);

    switch (method) {
    case BufferMethods::BindObject: {
        ProcessBindMethod(dma_pusher, method_call);
        break;
    }
    case BufferMethods::Nop:
    case BufferMethods::SemaphoreAddressHigh:
    case BufferMethods::SemaphoreAddressLow:
    case BufferMethods::SemaphoreSequencePayload:
    case BufferMethods::SyncpointPayload:
    case BufferMethods::WrcacheFlush:
        break;
    case BufferMethods::RefCnt:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->SignalReference(); },
               FenceWrites);
        break;
    case BufferMethods::SyncpointOperation:
        ProcessFenceActionMethod(dma_pusher);
        break;
    case BufferMethods::WaitForIdle:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->WaitForIdle(); },
               VideoCommon::GpuSplit::Writes::Caches);
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case BufferMethods::SemaphoreOperation: {
        ProcessSemaphoreTriggerMethod(dma_pusher);
        break;
    }
    case BufferMethods::NonStallInterrupt: {
        LOG_ERROR(HW_GPU, "Special puller engine method NonStallInterrupt not implemented");
        break;
    }
    case BufferMethods::MemOpA: {
        LOG_TRACE(HW_GPU, "Memory Operation A");
        break;
    }
    case BufferMethods::MemOpB: {
        // Implement this better.
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->InvalidateGPUCache(); },
               VideoCommon::GpuSplit::Writes::Caches);
        break;
    }
    case BufferMethods::MemOpC:
    case BufferMethods::MemOpD: {
        LOG_TRACE(HW_GPU, "Memory Operation C,D");
        break;
    }
    case BufferMethods::SemaphoreAcquire: {
        // On Ampere host classes 0x17..0x1A are SEM_ADDR_LO/HI and SEM_PAYLOAD_LO/HI.
        if (!g_nx2_ampere) {
            ProcessSemaphoreAcquire(dma_pusher);
        }
        break;
    }
    case BufferMethods::SemaphoreRelease: {
        if (g_nx2_ampere) {
            ProcessSemaphoreExecute(dma_pusher, method_call.argument);
        } else {
            ProcessSemaphoreRelease(dma_pusher);
        }
        break;
    }
    case BufferMethods::Yield: {
        // TODO(Kmather73): Research and implement this method.
        LOG_ERROR(HW_GPU, "Special puller engine method Yield not implemented");
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Special puller engine method {:X} not implemented", method);
        if (static bool dumped = false; g_nx2_ampere && !dumped) {
            dumped = true;
            dma_pusher.DumpCurrentEntry("unknown host method");
        }
        break;
    }
}

/// Calls a GPU engine method.
static bool Nx2MethodTrace() {
    static const bool trace = std::getenv("EDEN_NX2_METHOD_TRACE") != nullptr;
    return trace;
}

void Puller::CallEngineMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    const EngineID engine = bound_engines[method_call.subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 method sub={} engine={:04X} method={:#x} arg={:#x}",
                 method_call.subchannel, static_cast<u32>(engine), method_call.method,
                 method_call.argument);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU engine multivalue method.
void Puller::CallEngineMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    const EngineID engine = bound_engines[subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 multimethod sub={} engine={:04X} method={:#x} amount={} first={:#x}",
                 subchannel, static_cast<u32>(engine), method, amount, base_start[0]);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU method.
void Puller::CallMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method_call.method, method_call.subchannel);
    ASSERT(method_call.subchannel < bound_engines.size());

    if (ExecuteMethodOnEngine(dma_pusher, method_call.method)) {
        CallEngineMethod(dma_pusher, method_call);
    } else {
        CallPullerMethod(dma_pusher, method_call);
    }
}

/// Calls a GPU multivalue method.
void Puller::CallMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method, subchannel);
    ASSERT(subchannel < bound_engines.size());
    if (ExecuteMethodOnEngine(dma_pusher, method)) {
        CallEngineMultiMethod(dma_pusher, method, subchannel, base_start, amount, methods_pending);
    } else {
        for (u32 i = 0; i < amount; i++) {
            CallPullerMethod(dma_pusher, MethodCall{
                method,
                base_start[i],
                subchannel,
                methods_pending - i,
            });
        }
    }
}

/// Determines where the method should be executed.
[[nodiscard]] bool Puller::ExecuteMethodOnEngine(DmaPusher& dma_pusher, u32 method) {
    const auto buffer_method = BufferMethods(method);
    return buffer_method >= BufferMethods::NonPullerMethods;
}

} // namespace Tegra::Engines
