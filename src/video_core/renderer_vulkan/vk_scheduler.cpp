// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <optional>
#include <vector>
#include <string_view>
#include <string>
#include <map>
#include <tuple>
#include <mutex>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_query_cache.h"

#include "common/logging.h"
#include "common/settings.h"
#include "common/dkb_cpu_trace.h"
#include "common/thread.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/gpu_wait_stats.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_command_pool.h"
#include "video_core/renderer_vulkan/vk_gpu_cost.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/upload_batch.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

namespace {
std::atomic<u64> g_profile_submits{};
std::atomic<u64> g_profile_renderpasses{};
std::atomic<u64> g_profile_finishes{};
std::atomic<u64> g_profile_finish_ns{};
std::atomic<u64> g_profile_window_start{};
std::mutex g_profile_end_mutex;
std::map<std::tuple<const char*, u32, const char*>, u64> g_profile_ends;

void CountRenderPassEnd(const std::source_location& location) {
    if (!VideoCommon::Nx2Profile::Enabled()) {
        return;
    }
    std::scoped_lock lock{g_profile_end_mutex};
    ++g_profile_ends[{location.file_name(), location.line(), location.function_name()}];
}

void ReportSubmitCounts() {
    const u64 now = VideoCommon::Nx2Profile::Now();
    u64 start = g_profile_window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        g_profile_window_start.compare_exchange_strong(start, now);
        return;
    }
    if (now - start < 10'000'000'000ULL ||
        !g_profile_window_start.compare_exchange_strong(start, now)) {
        return;
    }
    const double seconds = static_cast<double>(now - start) / 1e9;
    LOG_INFO(Render_Vulkan,
             "Scheduler per second: {:.0f} submits, {:.0f} render passes, {:.0f} finishes "
             "waiting {:.1f} ms",
             static_cast<double>(g_profile_submits.exchange(0)) / seconds,
             static_cast<double>(g_profile_renderpasses.exchange(0)) / seconds,
             static_cast<double>(g_profile_finishes.exchange(0)) / seconds,
             static_cast<double>(g_profile_finish_ns.exchange(0)) / 1e6 / seconds);
    std::map<std::string, u64> named;
    {
        std::scoped_lock lock{g_profile_end_mutex};
        for (const auto& [site, count] : g_profile_ends) {
            const auto& [file, line, function] = site;
            named[GpuCost::SiteName(file, line, function)] += count;
        }
        g_profile_ends.clear();
    }
    std::vector<std::pair<u64, std::string>> ends;
    for (const auto& [site, count] : named) {
        ends.emplace_back(count, site);
    }
    std::sort(ends.rbegin(), ends.rend());
    for (size_t i = 0; i < ends.size() && i < 16; ++i) {
        LOG_INFO(Render_Vulkan, "Render pass ends {:.0f}/s at {}",
                 static_cast<double>(ends[i].first) / seconds, ends[i].second);
    }
}
}

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf,
                                         vk::CommandBuffer upload_cmdbuf, GpuCost* cost) {
    auto command = first;
    for (size_t index = 0; command != nullptr; ++index) {
        auto next = command->GetNext();
        if (cost != nullptr && index < sites.size()) [[unlikely]] {
            cost->BeginCommand(sites[index]);
        }
        command->Execute(cmdbuf, upload_cmdbuf);
        command->~Command();
        command = next;
    }
    sites.clear();
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Device& device_, StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_},
      master_semaphore{std::make_unique<MasterSemaphore>(device)},
      command_pool{std::make_unique<CommandPool>(*master_semaphore, device)},
      upload_batch{std::make_unique<UploadBatch>()},
      reordered_uploads{std::make_unique<ReorderedUploadBatch>()} {

    if (Common::DkbFrameTrace::Enabled()) {
        const auto physical = device.GetPhysical();
        trace_timestamp_bits =
            physical.GetQueueFamilyProperties()[device.GetGraphicsFamily()].timestampValidBits;
        trace_timestamp_period = physical.GetProperties().limits.timestampPeriod;
        if (trace_timestamp_bits != 0 && trace_timestamp_bits <= 64 && trace_timestamp_period > 0) {
            try {
                trace_query_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = TraceMaxSubmissions * 2,
                    .pipelineStatistics = 0,
                });
                trace_submissions.resize(TraceMaxSubmissions);
                trace_work_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = TraceMaxWork * 2,
                    .pipelineStatistics = 0,
                });
                trace_work_items.resize(TraceMaxWork);
            } catch (const vk::Exception& error) {
                LOG_WARNING(Render_Vulkan, "DKB gpu_timestamp unavailable: {}", error.what());
            }
        } else {
            LOG_INFO(Render_Vulkan, "DKB gpu_timestamp unsupported: valid_bits={} period_ns={}",
                     trace_timestamp_bits, trace_timestamp_period);
        }
    }
    if (VideoCommon::ReadTrace::Enabled()) {
        CreateReadTrace();
    }
    gpu_cost = GpuCost::Create(device, trace_timestamp_bits, trace_timestamp_period,
                               static_cast<bool>(trace_query_pool));
    record_sites = gpu_cost != nullptr;

    AcquireNewChunk();
    AllocateWorkerCommandBuffer();
    worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
}

Scheduler::~Scheduler() {
    if (trace_query_pool || read_trace) {
        WaitWorker();
        worker_thread.request_stop();
        worker_thread.join();
        if (trace_gpu_tick != 0) {
            master_semaphore->Wait(trace_gpu_tick);
        }
        // The read trace's query pools are in every submission.
        if (read_trace && CurrentTick() > 1) {
            master_semaphore->Wait(CurrentTick() - 1);
        }
    }
}

/// TEST (EDEN_NX2_READ_TRACE): device timestamps that place guest reads on the GPU's timeline.
struct Scheduler::ReadTraceState {
    /// Submissions timed at their start and end (two queries a slot), by tick.
    static constexpr u32 TickSlots = 4096;
    /// TraceMark timestamps, by handle.
    static constexpr u32 MarkSlots = 4096;

    vk::QueryPool ticks;
    vk::QueryPool marks;
    /// vkGetCalibratedTimestamps{KHR,EXT}; null without them.
    PFN_vkGetCalibratedTimestampsKHR calibrate{};
    double ns_per_tick{};
    u64 next_tick{1}; ///< Worker: the submission its command buffers go in
    u64 next_mark{1}; ///< Recording thread
    std::atomic<u64> issued_marks{};
    std::atomic<u64> submitted{}; ///< The last tick vkQueueSubmit returned for
    /// When vkQueueSubmit returned (Now()) for the tick in submit_ticks
    std::array<std::atomic<u64>, TickSlots> submit_ticks{};
    std::array<std::atomic<u64>, TickSlots> submit_times{};
};

void Scheduler::CreateReadTrace() {
    const auto physical = device.GetPhysical();
    const u32 valid_bits =
        physical.GetQueueFamilyProperties()[device.GetGraphicsFamily()].timestampValidBits;
    const float period = physical.GetProperties().limits.timestampPeriod;
    if (valid_bits == 0 || period <= 0.0f) {
        LOG_WARNING(Render_Vulkan, "Read trace: the graphics queue has no timestamps");
        return;
    }
    auto created = std::make_unique<ReadTraceState>();
    created->ns_per_tick = static_cast<double>(period);
    const auto create_pool = [this](u32 count) {
        return device.GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = count,
            .pipelineStatistics = 0,
        });
    };
    try {
        created->ticks = create_pool(ReadTraceState::TickSlots * 2);
        created->marks = create_pool(ReadTraceState::MarkSlots);
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "Read trace: no query pools: {}", error.what());
        return;
    }
    // The device enables either extension with EDEN_NX2_READ_TRACE (vulkan_device.cpp).
    const auto& dld = device.GetDispatchLoader();
    const auto& extensions = device.GetAvailableExtensions();
    const bool khr = extensions.contains(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    if (khr || extensions.contains(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
        const auto get_domains =
            reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                dld.vkGetInstanceProcAddr(device.GetInstance(),
                                          khr ? "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR"
                                              : "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
        const auto calibrate = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(
            dld.vkGetDeviceProcAddr(*device.GetLogical(), khr ? "vkGetCalibratedTimestampsKHR"
                                                              : "vkGetCalibratedTimestampsEXT"));
        std::array<VkTimeDomainKHR, 8> domains{};
        u32 count = static_cast<u32>(domains.size());
        if (get_domains != nullptr && calibrate != nullptr &&
            get_domains(physical, &count, domains.data()) >= VK_SUCCESS &&
            std::ranges::find(domains.begin(), domains.begin() + (std::min)(count, 8u),
                              VK_TIME_DOMAIN_DEVICE_KHR) !=
                domains.begin() + (std::min)(count, 8u)) {
            created->calibrate = calibrate;
        }
    }
    LOG_INFO(Render_Vulkan, "Read trace: device timestamps every {} ns, {} bits, {}", period,
             valid_bits,
             created->calibrate ? "device clock readable (calibrated timestamps)"
                                : "no calibrated timestamps: reads stay unplaced");
    read_trace = std::move(created);
}

u64 Scheduler::TraceMark() {
    if (!read_trace) {
        return 0;
    }
    const u64 handle = read_trace->next_mark++;
    read_trace->issued_marks.store(handle, std::memory_order_release);
    RecordWithUploadBuffer(
        [pool = *read_trace->marks, slot = static_cast<u32>(handle % ReadTraceState::MarkSlots)](
            vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
            // The upload buffer runs first in the same submission.
            upload_cmdbuf.ResetQueryPool(pool, slot, 1);
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, slot);
        });
    return handle;
}

void Scheduler::TraceRead(VideoCommon::ReadTrace::Read& read, VideoCommon::ReadTrace::Step step) {
    namespace ReadTrace = VideoCommon::ReadTrace;
    if (!read_trace) {
        return;
    }
    switch (step) {
    case ReadTrace::Step::Request:
        master_semaphore->Refresh();
        read.completed = master_semaphore->KnownGpuTick();
        read.submitted = read_trace->submitted.load(std::memory_order_acquire);
        read.recording = master_semaphore->CurrentTick();
        return;
    case ReadTrace::Step::Calibrate:
        if (read_trace->calibrate != nullptr) {
            const VkCalibratedTimestampInfoKHR info{
                .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
                .pNext = nullptr,
                .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR,
            };
            u64 value{};
            u64 deviation{};
            const u64 before = ReadTrace::Now();
            if (read_trace->calibrate(*device.GetLogical(), 1, &info, &value, &deviation) ==
                VK_SUCCESS) {
                const u64 after = ReadTrace::Now();
                read.device = value;
                read.host = before + (after - before) / 2;
                read.uncertainty = after - before;
            }
        }
        return;
    case ReadTrace::Step::Done:
        ReadTrace::Record(read, PlaceRead(read));
        return;
    }
}

VideoCommon::ReadTrace::Timeline Scheduler::PlaceRead(
    const VideoCommon::ReadTrace::Read& read) const {
    namespace ReadTrace = VideoCommon::ReadTrace;
    if (!read.has_copy) {
        return {};
    }
    const ReadTraceState& trace = *read_trace;
    const u64 copy_tick = read.copy.tick;
    struct Result {
        u64 value;
        u64 available;
    };
    const auto& logical = device.GetLogical();
    const auto mark = [&](u64 handle) -> std::optional<u64> {
        const auto reused = [&] {
            return trace.issued_marks.load(std::memory_order_acquire) - handle >=
                   ReadTraceState::MarkSlots;
        };
        Result result{};
        if (handle == 0 || reused() ||
            logical.GetQueryResults(
                *trace.marks, static_cast<u32>(handle % ReadTraceState::MarkSlots), 1,
                sizeof(result), &result, sizeof(result),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS ||
            result.available == 0 || reused()) {
            return std::nullopt;
        }
        return result.value;
    };
    // The submissions after the last one done at the request, up to the copy's: all done now.
    const auto place = [&]() -> ReadTrace::Timeline {
        const u64 first_tick = read.completed + 1;
        if (read.device == 0 || !read.has_write || copy_tick < first_tick ||
            copy_tick - first_tick >= ReadTraceState::TickSlots / 2) {
            return {};
        }
        const auto write_end = mark(read.write.mark);
        const auto copy_start = mark(read.copy.start);
        const auto copy_end = mark(read.copy.end);
        if (!write_end || !copy_start || !copy_end) {
            return {};
        }
        const u32 tick_count = static_cast<u32>(copy_tick - first_tick + 1);
        std::vector<Result> results(size_t{tick_count} * 2);
        for (u32 done = 0; done < tick_count;) {
            const u32 slot = static_cast<u32>((first_tick + done) % ReadTraceState::TickSlots);
            const u32 count = (std::min)(tick_count - done, ReadTraceState::TickSlots - slot);
            if (logical.GetQueryResults(
                    *trace.ticks, slot * 2, count * 2, size_t{count} * 2 * sizeof(Result),
                    &results[size_t{done} * 2], sizeof(Result),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS) {
                return {};
            }
            done += count;
        }
        std::vector<std::pair<u64, u64>> submissions;
        submissions.reserve(tick_count);
        for (u32 i = 0; i < tick_count; ++i) {
            const Result& start = results[size_t{i} * 2];
            const Result& end = results[size_t{i} * 2 + 1];
            if (start.available == 0 || end.available == 0) {
                return {};
            }
            submissions.emplace_back(start.value, end.value);
        }
        return ReadTrace::Place(read, ReadTrace::DeviceTimes{
                                          .ns_per_tick = trace.ns_per_tick,
                                          .write_end = *write_end,
                                          .copy_start = *copy_start,
                                          .copy_end = *copy_end,
                                          .submissions = submissions,
                                      });
    };
    ReadTrace::Timeline timeline = place();
    const u32 copy_slot = static_cast<u32>(copy_tick % ReadTraceState::TickSlots);
    if (trace.submit_ticks[copy_slot].load(std::memory_order_acquire) == copy_tick) {
        const u64 submitted_at = trace.submit_times[copy_slot].load(std::memory_order_relaxed);
        if (submitted_at >= read.copy.flushed) {
            timeline.has_submit = true;
            timeline.submit = submitted_at - read.copy.flushed;
        }
    }
    return timeline;
}

u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                     std::source_location location) {
    if (state.renderpass) {
        CountRenderPassEnd(location);
    }
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Flush");
    }
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore, location);
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    return signal_value;
}

bool Scheduler::DeferFlush() {
    // Fences come one per compute release: ~11,500 a second in Hyrule Warriors: Age of
    // Imprisonment fights, each a queue submission (~22,000 a second with the others), which
    // kept the Vulkan worker busy submitting. Batched within 500 us: ~10-15% more fps there.
    // Command list copies after dispatches were ~9,000 more a second, of which the front waited
    // for about 5%.
    static const s64 interval_us = [] {
        const char* value = std::getenv("EDEN_NX2_FENCE_FLUSH_US");
        if (value != nullptr) {
            return static_cast<s64>(std::atoll(value));
        }
        return s64{500};
    }();
    if (interval_us <= 0 || !VideoCommon::GpuSplit::OnBackThread() ||
        std::chrono::steady_clock::now() - last_flush >= std::chrono::microseconds{interval_us}) {
        return false;
    }
    flush_pending = true;
    return true;
}

void Scheduler::FlushDeferred() {
    if (flush_pending) {
        Flush();
    }
}

bool Scheduler::WaitSubmitted(u64 tick) {
    if (IsSubmitted(tick)) {
        return true;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    split.RequestIdleHook();
    // No time limit: the back can spend seconds on one entry (a large shader's translation),
    // and a fence reported as submitted before it is would release its downloads too early.
    const auto start = std::chrono::steady_clock::now();
    auto next_log = start + std::chrono::seconds{2};
    for (u32 spins = 0; !IsSubmitted(tick); ++spins) {
        if (!split.BackRunning()) {
            return IsSubmitted(tick);
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_log) {
            next_log = now + std::chrono::seconds{2};
            LOG_WARNING(Render_Vulkan, "Waited {} s for the GPU thread to submit tick {}",
                        std::chrono::duration_cast<std::chrono::seconds>(now - start).count(),
                        tick);
            split.RequestIdleHook();
        }
        if (spins < 64) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
    }
    return true;
}

void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                       std::source_location location) {
    if (state.renderpass) {
        CountRenderPassEnd(location);
    }
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Finish");
    }
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    const bool profile = VideoCommon::Nx2Profile::Enabled();
    const u64 start = profile ? VideoCommon::Nx2Profile::Now() : 0;
    const u64 wait_start = VideoCommon::GpuWaitStats::Now();
    SubmitExecution(signal_semaphore, wait_semaphore, location);
    Wait(presubmit_tick);
    AllocateNewContext();
    VideoCommon::GpuWaitStats::AddHostWait(VideoCommon::GpuWaitStats::Now() - wait_start);
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    if (profile) {
        g_profile_finishes.fetch_add(1, std::memory_order_relaxed);
        g_profile_finish_ns.fetch_add(VideoCommon::Nx2Profile::Now() - start,
                                      std::memory_order_relaxed);
    }
}

void Scheduler::WaitWorker() {
    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    std::scoped_lock el{execution_mutex};
}

void Scheduler::MarkFrameEnd(u64 frame) {
    if (Common::DkbFrameTrace::Enabled()) {
        Record([this, frame](vk::CommandBuffer) { trace_frame_end = frame; });
    }
}

void Scheduler::DispatchWork() {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::DispatchWork");
    }
    if (uploads_pending) {
        // What the worker gets is everything recorded so far.
        FlushUploads();
    }
    if (chunk && !chunk->Empty()) {
        // Waking the worker is a system call (~2% of the GPU thread at one dispatch per 8
        // draws): skip it while the worker is still busy, it pops the queue before waiting.
        bool wake;
        {
            std::scoped_lock ql{queue_mutex};
            work_queue.push(std::move(chunk));
            wake = worker_waiting;
        }
        if (wake) {
            event_cv.notify_all();
        }
        AcquireNewChunk();
    }
}

void Scheduler::RecordUpload(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> copies,
                             std::source_location location) {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::RecordUpload");
    }
    if (copies.empty()) {
        return;
    }
    if (uploads_pending && !upload_batch->CanAdd(src, dst, copies)) {
        FlushUploads();
    }
    if (!uploads_pending) {
        upload_batch_site = location;
    }
    upload_batch->Add(src, dst, copies);
    uploads_pending = true;
}

void Scheduler::FlushUploads() {
    // Copies, so the batch's vectors keep their capacity for the next one.
    auto command = [copies = upload_batch->copies,
                    groups = upload_batch->groups](vk::CommandBuffer cmdbuf, vk::CommandBuffer) {
        UploadBatch::Record(cmdbuf, copies, groups);
    };
    upload_batch->copies.clear();
    upload_batch->groups.clear();
    uploads_pending = false;
    RecordAs(std::move(command), upload_batch_site, RecordSite::Own, 0);
}

void Scheduler::RecordReorderedUpload(VkBuffer src, VkBuffer dst,
                                      std::span<const VkBufferCopy> copies, u64 prior_tick) {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::RecordReorderedUpload");
    }
    if (copies.empty()) {
        return;
    }
    reordered_uploads->Add(src, dst, copies, prior_tick);
    reordered_uploads_pending = true;
}

void Scheduler::FlushReorderedUploads() {
    std::vector<VkBufferCopy> batch_copies;
    std::vector<UploadBatch::Group> batch_groups;
    u64 prior_tick = 0;
    reordered_uploads->Take(batch_copies, batch_groups, prior_tick);
    reordered_uploads_pending = false;
    // The copies must come after earlier submissions' work on their bytes: finished already, or
    // waited for by a barrier. The host GPU is asked once a submission, when it may still run it.
    bool batch_after_prior = prior_tick != 0 && !master_semaphore->IsFree(prior_tick);
    if (batch_after_prior) {
        master_semaphore->Refresh();
        batch_after_prior = !master_semaphore->IsFree(prior_tick);
    }
    RecordAs(
        [copies = std::move(batch_copies), groups = std::move(batch_groups),
         after_prior_work = batch_after_prior](vk::CommandBuffer, vk::CommandBuffer upload_cmdbuf) {
            ReorderedUploadBatch::Record(upload_cmdbuf, copies, groups, after_prior_work);
        },
        std::source_location::current(), RecordSite::Own, 0);
}

void Scheduler::BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                                    const VkClearValue* clear_values, u32 clear_value_count) {
    if (VideoCommon::Nx2Profile::Enabled()) {
        g_profile_renderpasses.fetch_add(1, std::memory_order_relaxed);
    }
    ++trace_counts.render_passes;
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    ++render_pass_serial;
    state.renderpass = renderpass;
    state.framebuffer = framebuffer_handle;
    state.render_area = render_area;

    if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string render_pass_info =
            fmt::format("renderArea={}x{}, numImages={}", render_area.width, render_area.height,
                        framebuffer->NumImages());
        GPU::Logging::GPULogger::GetInstance().LogRenderPassBegin(render_pass_info);
    }

    std::array<VkClearValue, 9> values{};
    for (u32 i = 0; i < clear_value_count && i < values.size(); ++i) {
        values[i] = clear_values[i];
    }
    const bool after_compute = std::exchange(compute_writes_pending, false);
    RecordAs([renderpass, framebuffer_handle, render_area, values, clear_value_count,
              after_compute](vk::CommandBuffer cmdbuf, vk::CommandBuffer) {
        const VkRenderPassBeginInfo renderpass_bi{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = renderpass,
            .framebuffer = framebuffer_handle,
            .renderArea =
                {
                    .offset = {.x = 0, .y = 0},
                    .extent = render_area,
                },
            .clearValueCount = clear_value_count,
            .pClearValues = clear_value_count != 0 ? values.data() : nullptr,
        };
        if (after_compute) {
            // Draws read compute output as indirect arguments, vertices and resources; nothing
            // else orders those reads after the dispatch.
            static constexpr VkMemoryBarrier COMPUTE_TO_GRAPHICS{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, COMPUTE_TO_GRAPHICS);
        }
        cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
    }, last_pass_end, RecordSite::PassBegin, 0);
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
    framebuffer->MarkResolveShadowsUpToDate();
}

void Scheduler::RealizeDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const DeferredClear dc = deferred_clear;
    deferred_clear = {};

    std::array<VkClearValue, 9> clear_values{};
    u32 count = 0;
    const RenderPassKey& base = dc.framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (base.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            continue;
        }
        clear_values[count++] = dc.color_values[slot];
    }
    if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        clear_values[count++] = dc.depth_stencil_value;
    }
    const u32 color_discard_mask =
        dc.framebuffer->DiscardsMsaaColor() ? dc.color_clear_mask : 0u;
    const bool depth_stencil_discard =
        dc.depth_stencil && dc.framebuffer->DiscardsMsaaDepthStencil();
    const VkRenderPass renderpass = dc.framebuffer->RenderPassVariant(
        dc.color_clear_mask, dc.depth_stencil, color_discard_mask, depth_stencil_discard);
    EndRenderPass();
    BeginRenderPassImpl(dc.framebuffer, renderpass, clear_values.data(), count);
}

bool Scheduler::DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot,
                                const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.color_clear_mask |= 1u << rt_slot;
    deferred_clear.color_values[rt_slot] = value;
    return true;
}

bool Scheduler::DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.depth_stencil = true;
    deferred_clear.depth_stencil_value = value;
    return true;
}

void Scheduler::FlushDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    RealizeDeferredClear();
    EndRenderPass();
}

void Scheduler::RequestRenderpass(const Framebuffer* framebuffer, std::source_location location) {
    if (deferred_clear.framebuffer == framebuffer) {
        RealizeDeferredClear();
        return;
    }
    const VkRenderPass renderpass = framebuffer->RenderPass();
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    // A draw continues the active pass whenever the same framebuffer and area are bound. The
    // render-pass *variant* (a deferred clear's loadOp=CLEAR form) is only meaningful at begin -
    // loadOps are applied once, and the base/variant passes are render-pass-compatible (only
    // loadOp/storeOp differ). Comparing the exact handle here made every draw after a deferred
    // clear restart the pass and re-LOAD what was just CLEARed (~118 MB of attachment traffic per
    // world-target turn). The framebuffer handle already fixes the attachment structure and key,
    // so it is the correct identity to compare.
    if (state.renderpass && framebuffer_handle == state.framebuffer &&
        render_area.width == state.render_area.width &&
        render_area.height == state.render_area.height) {
        return;
    }
    if (state.renderpass) {
        CountRenderPassEnd(location);
    }
    // Ends any active pass and realizes a deferred clear
    EndRenderPass(location);
    BeginRenderPassImpl(framebuffer, renderpass, nullptr, 0);
}

void Scheduler::RequestOutsideRenderPassOperationContext(std::source_location location) {
    if (state.renderpass) {
        CountRenderPassEnd(location);
    }
    EndRenderPass(location);
}

bool Scheduler::UpdateGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (state.graphics_pipeline == pipeline) {
        if (pipeline && pipeline->UsesExtendedDynamicState() &&
            state.needs_state_enable_refresh) {
            state_tracker.InvalidateStateEnableFlag();
            state.needs_state_enable_refresh = false;
        }
        return false;
    }

    state.graphics_pipeline = pipeline;

    if (!pipeline) {
        return true;
    }

    if (!pipeline->UsesExtendedDynamicState()) {
        state.needs_state_enable_refresh = true;
    } else if (state.needs_state_enable_refresh) {
        state_tracker.InvalidateStateEnableFlag();
        state.needs_state_enable_refresh = false;
    }

    return true;
}

bool Scheduler::UpdateRescaling(bool is_rescaling, bool mesh_layout) {
    if (state.rescaling_defined && is_rescaling == state.is_rescaling &&
        mesh_layout == state.mesh_layout) {
        return false;
    }
    state.rescaling_defined = true;
    state.is_rescaling = is_rescaling;
    state.mesh_layout = mesh_layout;
    return true;
}

bool Scheduler::UpdateDescriptorBufferChunk(u32 descriptor_chunk) {
    if (state.descriptor_buffer_bound && descriptor_chunk == state.descriptor_buffer_chunk) {
        return false;
    }
    state.descriptor_buffer_bound = true;
    state.descriptor_buffer_chunk = descriptor_chunk;
    return true;
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");
    const Common::DkbCpuTrace::Registration trace_thread{Common::DkbCpuTrace::Role::VulkanWorker};
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            event_cv.wait(lk, stop_token, [&] {
                worker_waiting = !TryPopQueue(work);
                return !worker_waiting;
            });
            worker_waiting = false;

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            void(std::exchange(lk, std::unique_lock{execution_mutex}));

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            work->ExecuteAll(current_cmdbuf, current_upload_cmdbuf,
                             trace_gpu_recording ? gpu_cost.get() : nullptr);

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffer();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffer() {
    current_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_upload_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_upload_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    if (read_trace) {
        // When the host GPU starts the submission these go in (its end: SubmitExecution).
        const u32 query = static_cast<u32>(read_trace->next_tick % ReadTraceState::TickSlots) * 2;
        current_upload_cmdbuf.ResetQueryPool(*read_trace->ticks, query, 2);
        current_upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *read_trace->ticks,
                                             query);
    }
    if (trace_query_pool && !trace_gpu_failed) {
        TraceGpuSubmission();
        if (gpu_cost && trace_gpu_recording) {
            gpu_cost->BeginSubmission(current_cmdbuf, current_upload_cmdbuf, trace_gpu_count == 0);
        }
    }
}

void Scheduler::TraceShader(u64 hash, u32 stage) {
    if (hash == 0) {
        return;
    }
    for (u32 i = 0; i < trace_work.num_shaders; ++i) {
        if (trace_work.shaders[i].hash == hash && trace_work.shaders[i].stage == stage) {
            return;
        }
    }
    if (trace_work.num_shaders == trace_work.shaders.size()) {
        ++trace_work.untracked_uses;
        return;
    }
    trace_work.shaders[trace_work.num_shaders++] = {hash, stage};
}

void Scheduler::TraceGraphicsPipeline(const std::array<u64, 6>& hashes) {
    if (!trace_query_pool) {
        return;
    }
    Record([this, hashes](vk::CommandBuffer) {
        if (!trace_gpu_recording) {
            return;
        }
        ++trace_work.draw_attempts;
        for (u32 stage = 0; stage < hashes.size(); ++stage) {
            TraceShader(hashes[stage], stage);
        }
    });
}

void Scheduler::TraceComputePipeline(u64 hash) {
    if (!trace_query_pool) {
        return;
    }
    Record([this, hash](vk::CommandBuffer) {
        if (!trace_gpu_recording) {
            return;
        }
        ++trace_work.dispatch_attempts;
        TraceShader(hash, 6);
    });
}

void Scheduler::ReportWorkTimes() {
    const u32 count = trace_work_count;
    trace_work_count = 0;
    if (count == 0) {
        return;
    }
    struct Result {
        u64 timestamp;
        u64 available;
    };
    std::vector<Result> results(count * 2);
    const auto result = device.GetLogical().GetQueryResults(
        *trace_work_pool, 0, count * 2, results.size() * sizeof(Result), results.data(),
        sizeof(Result), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS) {
        return;
    }
    if (read_trace && count < TraceMaxWork) {
        // Every draw and dispatch of the frame was timed: the reads placed in it split by them.
        std::vector<std::pair<u64, u64>> work;
        work.reserve(count);
        for (u32 i = 0; i < count; ++i) {
            if (results[i * 2].available != 0 && results[i * 2 + 1].available != 0) {
                work.emplace_back(results[i * 2].timestamp, results[i * 2 + 1].timestamp);
            }
        }
        VideoCommon::ReadTrace::NoteTimedWork(work, read_trace->ns_per_tick);
    }
    struct Totals {
        double total_ms{};
        double max_ms{};
        u32 count{};
        std::array<u32, 3> max_size{};
    };
    std::map<std::pair<u64, u64>, Totals> by_shaders;
    double total_ms{};
    for (u32 i = 0; i < count; ++i) {
        if (results[i * 2].available == 0 || results[i * 2 + 1].available == 0) {
            continue;
        }
        const double ms = static_cast<double>(Common::DkbFrameTrace::TimestampDelta(
                              results[i * 2].timestamp, results[i * 2 + 1].timestamp,
                              trace_timestamp_bits)) *
                          trace_timestamp_period / 1e6;
        const WorkTrace& item = trace_work_items[i];
        Totals& totals = by_shaders[{item.hash, item.fragment_hash}];
        totals.total_ms += ms;
        ++totals.count;
        if (ms > totals.max_ms) {
            totals.max_ms = ms;
            totals.max_size = item.size;
        }
        total_ms += ms;
    }
    std::vector<std::pair<std::pair<u64, u64>, Totals>> sorted(by_shaders.begin(),
                                                               by_shaders.end());
    std::ranges::sort(sorted, std::greater<>{},
                      [](const auto& entry) { return entry.second.total_ms; });
    if (sorted.size() > 8) {
        sorted.resize(8);
    }
    std::string slowest;
    for (const auto& [shaders, totals] : sorted) {
        const auto& [hash, fragment_hash] = shaders;
        slowest += fragment_hash == 0
                       ? fmt::format(" s6:{:016x}", hash)
                       : fmt::format(" s1:{:016x}+s5:{:016x}", hash, fragment_hash);
        slowest += fmt::format("={:.3f}ms/{}x(max {:.3f}ms size {}x{}x{})", totals.total_ms,
                               totals.count, totals.max_ms, totals.max_size[0],
                               totals.max_size[1], totals.max_size[2]);
    }
    LOG_INFO(Render_Vulkan,
             "DKB gpu_work renderer_frame={} timed={} timed_sum_ms={:.3f} slowest:{}",
             trace_sample_frame, count, total_ms, slowest);
}

void Scheduler::TraceGpuSubmission() {
    if (trace_gpu_pending) {
        master_semaphore->Refresh();
        if (!master_semaphore->IsFree(trace_gpu_tick)) {
            return;
        }
        struct Result {
            u64 timestamp;
            u64 available;
        };
        std::array<Result, TraceMaxSubmissions * 2> results{};
        const auto result = device.GetLogical().GetQueryResults(
            *trace_query_pool, 0, trace_gpu_count * 2, trace_gpu_count * 2 * sizeof(Result),
            results.data(), sizeof(Result),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_NOT_READY) {
            return;
        }
        if (result != VK_SUCCESS) {
            LOG_WARNING(Render_Vulkan, "DKB gpu_timestamp disabled: query result {}",
                        static_cast<int>(result));
            trace_gpu_failed = true;
            return;
        }
        double total_ms{};
        std::array<std::pair<double, u32>, 4> slowest{};
        for (u32 i = 0; i < trace_gpu_count * 2; i += 2) {
            if (results[i].available == 0 || results[i + 1].available == 0) {
                return;
            }
            const auto ticks = Common::DkbFrameTrace::TimestampDelta(
                results[i].timestamp, results[i + 1].timestamp, trace_timestamp_bits);
            const double duration_ms = static_cast<double>(ticks) * trace_timestamp_period / 1e6;
            total_ms += duration_ms;
            if (duration_ms >= 10.0 && duration_ms > slowest.back().first) {
                slowest.back() = {duration_ms, i / 2};
                std::ranges::sort(slowest, std::greater<>{});
            }
        }
        const double span_ms = trace_gpu_skipped == 0
            ? static_cast<double>(Common::DkbFrameTrace::TimestampDelta(
                  results[0].timestamp, results[trace_gpu_count * 2 - 1].timestamp,
                  trace_timestamp_bits)) * trace_timestamp_period / 1e6
            : -1.0;
        LOG_INFO(Render_Vulkan,
                 "DKB gpu_frame renderer_frame={} last_submission_tick={} "
                 "gpu_frame_span_ms={:.3f} gpu_submission_sum_ms={:.3f} "
                 "measured_submissions={} dropped_submissions={} "
                 "valid_bits={} period_ns={}",
                 trace_sample_frame, trace_gpu_tick, span_ms, total_ms, trace_gpu_count,
                 trace_gpu_skipped, trace_timestamp_bits, trace_timestamp_period);
        for (const auto& [duration_ms, index] : slowest) {
            if (duration_ms < 10.0) {
                continue;
            }
            const auto& work = trace_submissions[index];
            std::string shaders;
            for (u32 i = 0; i < work.num_shaders; ++i) {
                const auto& shader = work.shaders[i];
                shaders += fmt::format("{}s{}:{:016x}", shaders.empty() ? "" : ",",
                                       shader.stage, shader.hash);
            }
            LOG_INFO(Render_Vulkan,
                     "DKB gpu_submission renderer_frame={} tick={} gpu_ms={:.3f} "
                     "draw_attempts={} dispatch_attempts={} shaders={} untracked_uses={}",
                     trace_sample_frame, work.tick, duration_ms, work.draw_attempts,
                     work.dispatch_attempts, shaders, work.untracked_uses);
        }
        ReportWorkTimes();
        if (gpu_cost) {
            gpu_cost->FrameDone();
        }
        trace_gpu_tick = 0;
        trace_gpu_count = 0;
        trace_gpu_skipped = 0;
        trace_gpu_pending = false;
        trace_gpu_rate.next = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    }
    if (!trace_gpu_sampling || trace_gpu_count == TraceMaxSubmissions) {
        return;
    }
    current_upload_cmdbuf.ResetQueryPool(*trace_query_pool, trace_gpu_count * 2, 2);
    if (trace_gpu_count == 0 && trace_work_pool) {
        // Draws are timed inside render passes, where queries cannot be reset.
        current_upload_cmdbuf.ResetQueryPool(*trace_work_pool, 0, TraceMaxWork * 2);
    }
    current_upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *trace_query_pool,
                                        trace_gpu_count * 2);
    trace_gpu_recording = true;
}

u64 Scheduler::SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                                std::source_location cause) {
    if (uploads_pending) {
        // In the tick they were recorded at.
        FlushUploads();
    }
    if (VideoCommon::Nx2Profile::Enabled()) {
        g_profile_submits.fetch_add(1, std::memory_order_relaxed);
        ReportSubmitCounts();
    }
    EndPendingOperations(cause);
    InvalidateState();
    if (reordered_uploads_pending) {
        // Into the upload command buffer of the submission their uploads were decided for.
        FlushReorderedUploads();
    }

    const u64 signal_value = master_semaphore->NextTick();
    RecordAs([signal_semaphore, wait_semaphore, signal_value,
              this](vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        upload_cmdbuf.End();
        if (trace_gpu_recording) {
            if (gpu_cost) {
                gpu_cost->EndSubmission(cmdbuf);
            }
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_query_pool,
                                 trace_gpu_count * 2 + 1);
            trace_gpu_tick = signal_value;
            trace_work.tick = signal_value;
            trace_submissions[trace_gpu_count] = trace_work;
            ++trace_gpu_count;
            trace_gpu_recording = false;
        } else if (trace_gpu_sampling) {
            ++trace_gpu_skipped;
        }
        if (trace_query_pool) {
            trace_work = {};
        }
        if (read_trace) {
            const u32 query = static_cast<u32>(signal_value % ReadTraceState::TickSlots) * 2 + 1;
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *read_trace->ticks, query);
        }
        cmdbuf.End();

        if (on_submit) {
            on_submit();
        }

        std::scoped_lock lock{submit_mutex};
        switch (const VkResult result = master_semaphore->SubmitQueue(
                    cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore, signal_value)) {
        case VK_SUCCESS:
            // Log successful queue submission
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_vulkan_calls.GetValue()) {
                GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                    "vkQueueSubmit", "", VK_SUCCESS);
            }
            break;
        case VK_ERROR_DEVICE_LOST:
            device.ReportLoss();
            [[fallthrough]];
        default:
            vk::Check(result);
            break;
        }
        if (read_trace) {
            const u32 slot = static_cast<u32>(signal_value % ReadTraceState::TickSlots);
            read_trace->submit_times[slot].store(VideoCommon::ReadTrace::Now(),
                                                 std::memory_order_relaxed);
            read_trace->submit_ticks[slot].store(signal_value, std::memory_order_release);
            read_trace->submitted.store(signal_value, std::memory_order_release);
            read_trace->next_tick = signal_value + 1;
        }
        if (trace_frame_end != 0) {
            if (trace_gpu_sampling) {
                trace_gpu_pending = true;
                trace_gpu_sampling = false;
            }
            if (trace_query_pool && !trace_gpu_failed && !trace_gpu_pending &&
                trace_gpu_rate.Ready(std::chrono::steady_clock::now())) {
                trace_sample_frame = trace_frame_end + 1;
                trace_gpu_sampling = true;
            }
            trace_frame_end = 0;
        }
    }, cause, RecordSite::Submit, 0);
    chunk->MarkSubmit();
    DispatchWork();
    return signal_value;
}

void Scheduler::AllocateNewContext() {
    // Enable counters once again. These are disabled when a command buffer is finished.
}

void Scheduler::InvalidateState() {
    ++recording_epoch;
    state.graphics_pipeline = nullptr;
    state.rescaling_defined = false;
    state.descriptor_buffer_bound = false;
    state_tracker.InvalidateCommandBufferState();
}

void Scheduler::EndPendingOperations(std::source_location cause) {
    // Pause, don't reset: a submission can land between a guest's counter reset and its report,
    // and resetting here would drop the samples counted so far (the report then reads too low).
    query_cache->CounterClose(VideoCommon::QueryType::ZPassPixelCount64);
    EndRenderPass(cause);
}

void Scheduler::EndRenderPass(std::source_location cause)
    {
        RealizeDeferredClear();
        if (!state.renderpass) {
            return;
        }
        ++recording_epoch;
        fragment_barrier.Reset();
        const u8 site_flags =
            static_cast<u8>(render_pass_has_stores ? RecordSite::PassHadStores : 0);
        render_pass_has_stores = false;
        last_pass_end = cause;
        ++render_pass_serial;

        query_cache->CounterClose(VideoCommon::QueryType::StreamingByteCount);

        // Log render pass end
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogRenderPassEnd();
        }

        query_cache->CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, false);
        query_cache->NotifySegment(false);

        RecordAs([num_images = num_renderpass_images,
                       images = renderpass_images,
                       ranges = renderpass_image_ranges,
                       has_transform_feedback = device.IsExtTransformFeedbackSupported()](
                          vk::CommandBuffer cmdbuf, vk::CommandBuffer) {
            std::array<VkImageMemoryBarrier, 9> barriers;
            for (size_t i = 0; i < num_images; ++i) {
                const VkImageSubresourceRange& range = ranges[i];
                const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const bool is_depth_stencil = (range.aspectMask
                                              & (VK_IMAGE_ASPECT_DEPTH_BIT
                                                 | VK_IMAGE_ASPECT_STENCIL_BIT)) !=0;

                VkAccessFlags src_access = 0;

                if (is_color)
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (is_depth_stencil)
                    src_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                else
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                barriers[i] = VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = src_access,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = images[i],
                        .subresourceRange = range,
                };
            }
            cmdbuf.EndRenderPass();
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, nullptr, nullptr, vk::Span(barriers.data(), num_images));
            if (has_transform_feedback) {
                static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                    .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, XFB_OUTPUT_BARRIER);
            }
        }, cause, RecordSite::PassEnd, site_flags);

        state.renderpass = VkRenderPass{};
        num_renderpass_images = 0;
    }


void Scheduler::AcquireNewChunk() {
    std::scoped_lock rl{reserve_mutex};

    if (chunk_reserve.empty()) {
        // If we don't have anything reserved, we need to make a new chunk.
        chunk = std::make_unique<CommandChunk>();
    } else {
        // Otherwise, we can just take from the reserve.
        chunk = std::move(chunk_reserve.back());
        chunk_reserve.pop_back();
    }
}

} // namespace Vulkan
