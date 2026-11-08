// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <queue>
#include <source_location>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "common/polyfill_thread.h"
#include "common/settings.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_read_trace.h"
#include "video_core/renderer_vulkan/fragment_barrier_state.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCommon {
template <typename Trait>
class QueryCacheBase;
}

namespace Vulkan {

class CommandPool;
class Device;
class Framebuffer;
class GpuCost;
class GraphicsPipeline;
class ReorderedUploadBatch;
class StateTracker;
class UploadBatch;

struct QueryCacheParams;

/// TEST (EDEN_NX2_GPU_COST, vk_gpu_cost.h): the source line that recorded a command. A render
/// pass's end carries the line that ended it, its begin the line that ended the pass before.
struct RecordSite {
    enum Role : u8 {
        Own,
        PassEnd,
        PassBegin,
        Submit,
    };
    enum Flags : u8 {
        PassHadStores = 1 << 0, ///< PassEnd: a draw in the pass wrote storage buffers or images
    };
    const char* file;
    const char* function;
    u32 line;
    Role role;
    u8 flags;
};

/// The scheduler abstracts command buffer and fence management with an interface that's able to do
/// OpenGL-like operations on Vulkan command buffers.
class Scheduler {
public:
    explicit Scheduler(const Device& device, StateTracker& state_tracker);
    ~Scheduler();

    /// Sends the current execution context to the GPU.
    u64 Flush(VkSemaphore signal_semaphore = {}, VkSemaphore wait_semaphore = {},
              std::source_location location = std::source_location::current());

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish(VkSemaphore signal_semaphore = {}, VkSemaphore wait_semaphore = {},
                std::source_location location = std::source_location::current());

    /// On the split GPU thread, a fence or a command list copy submits only if the last
    /// submission is older than a window (EDEN_NX2_FENCE_FLUSH_US, default 500 us, 0 submits at
    /// once); otherwise it goes with the next one. The back submits deferred work before it
    /// sleeps and when another thread asks (WaitSubmitted). True when the flush was deferred.
    bool DeferFlush();
    /// Submits deferred work (the split's back, from its idle hook).
    void FlushDeferred();
    /// Threads other than the split's back: waits until tick is submitted, asking the back to
    /// submit deferred work (logged every 2 s while it waits). False only when the back thread
    /// is not running, so nothing will submit it: the caller then submits it itself.
    bool WaitSubmitted(u64 tick);
    /// Whether everything up to tick has been handed to the worker for submission.
    [[nodiscard]] bool IsSubmitted(u64 tick) const noexcept {
        return tick < master_semaphore->CurrentTick();
    }

    /// Waits for the worker thread to finish executing everything. After this function returns it's
    /// safe to touch worker resources.
    void WaitWorker();

    void MarkFrameEnd(u64 frame);
    void TraceGraphicsPipeline(const std::array<u64, 6>& hashes);
    void TraceComputePipeline(u64 hash);
    /// Worker thread, inside a recorded command: timestamps one draw or dispatch of a sampled
    /// submission. A draw passes its vertex and fragment hashes, a dispatch its compute hash.
    template <typename Func>
    void TraceGpuWork(vk::CommandBuffer cmdbuf, u64 hash, u64 fragment_hash,
                      std::array<u32, 3> size, Func&& work) {
        if (!trace_gpu_recording || !trace_work_pool || trace_work_count == TraceMaxWork) {
            work();
            return;
        }
        const u32 query = trace_work_count * 2;
        trace_work_items[trace_work_count++] = {hash, fragment_hash, size};
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_work_pool, query);
        work();
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_work_pool, query + 1);
    }

    /// TEST (EDEN_NX2_READ_TRACE, nx2_read_trace.h), recording thread: a device timestamp written
    /// once the work recorded so far is done. Returns its handle for TraceRead, 0 when off.
    u64 TraceMark();
    /// Recording thread: the draws, dispatches and render passes recorded so far.
    [[nodiscard]] VideoCommon::ReadTrace::Counts TraceCounts() const noexcept {
        return trace_counts;
    }
    void TraceCountWork(bool dispatch) noexcept {
        ++(dispatch ? trace_counts.dispatches : trace_counts.draws);
    }
    /// A reading thread's step of a guest read that waits.
    void TraceRead(VideoCommon::ReadTrace::Read& read, VideoCommon::ReadTrace::Step step);

    /// Sends currently recorded work to the worker thread.
    void DispatchWork();

    /// Requests to begin a renderpass.
    void RequestRenderpass(const Framebuffer* framebuffer,
                           std::source_location location = std::source_location::current());

    /// Defers a full-attachment color clear so it becomes the next render pass.
    bool DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot, const VkClearValue& value);

    /// Defers a full depth/stencil clear so it becomes the next render pass.
    bool DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value);

    /// Realizes any pending deferred clear before its framebuffer can be moved or freed.
    void FlushDeferredClear();

    /// Requests the current execution context to be able to execute operations only allowed outside
    /// of a renderpass.
    void RequestOutsideRenderPassOperationContext(
        std::source_location location = std::source_location::current());

    void DeferFragmentBarrier() {
        fragment_barrier.Defer(state.renderpass != VK_NULL_HANDLE);
    }

    bool HasFragmentBarrier() const {
        return fragment_barrier.Pending();
    }

    void NoteRenderPassStores() {
        render_pass_has_stores = true;
    }

    bool RenderPassHasStores() const {
        return render_pass_has_stores;
    }

    /// A dispatch was recorded: the next render pass must see what it wrote.
    void NoteComputeWrites() {
        compute_writes_pending = true;
    }

    /// Returns true when a render pass is currently active in the scheduler state.
    bool IsRenderPassActive() const {
        return state.renderpass != VK_NULL_HANDLE;
    }

    /// Changes whenever a render pass begins or ends.
    u64 RenderPassSerial() const {
        return render_pass_serial;
    }

    /// The graphics pipeline bound in the current command buffer, if the scheduler knows it.
    const GraphicsPipeline* BoundGraphicsPipeline() const {
        return state.graphics_pipeline;
    }

    /// Update the pipeline to the current execution context.
    bool UpdateGraphicsPipeline(GraphicsPipeline* pipeline);

    /// Worker thread only: whether the bound graphics pipeline failed to build, so the draws
    /// recorded after it must be dropped instead of reaching the driver without a pipeline.
    void SetGraphicsPipelineMissing(bool missing) {
        graphics_pipeline_missing = missing;
    }
    /// The pipeline replaced its driver object (fast first build): bind it again at the next draw.
    void ForgetGraphicsPipeline(GraphicsPipeline* pipeline) {
        if (state.graphics_pipeline == pipeline) {
            state.graphics_pipeline = nullptr;
        }
    }
    [[nodiscard]] bool GraphicsPipelineMissing() const {
        return graphics_pipeline_missing;
    }

    /// Update the rescaling state. Returns true if the state has to be updated.
    bool UpdateRescaling(bool is_rescaling, bool mesh_layout);

    /// Returns true when the descriptor buffer chunk has to be bound into the command buffer.
    bool UpdateDescriptorBufferChunk(u32 descriptor_chunk);

    /// Invalidates current command buffer state except for render passes
    void InvalidateState();

    /// Assigns the query cache.
    void SetQueryCache(VideoCommon::QueryCacheBase<QueryCacheParams>& query_cache_) {
        query_cache = &query_cache_;
    }

    // Registers a callback to perform on queue submission.
    void RegisterOnSubmit(std::function<void()>&& func) {
        on_submit = std::move(func);
    }

    /// An upload in order: copies into dst after the work recorded before them and before the
    /// work recorded after them (the barriers of an in-order buffer copy). Uploads with no other
    /// command between them share one command and one pair of barriers, unless one would write
    /// bytes another writes, or a buffer another reads. Call outside a render pass.
    void RecordUpload(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> copies,
                      std::source_location location = std::source_location::current());

    /// An upload ahead of all the work of this submission's main command buffer: it runs in the
    /// upload command buffer, which the submission's reordered uploads are recorded into as one
    /// command when it is submitted. The caller guarantees that no work recorded in this
    /// submission uses the bytes it writes, nor work of an earlier submission after prior_tick
    /// (0: of any). Unless the host GPU has finished submission prior_tick by then, the upload
    /// command buffer begins with a barrier on the work of earlier submissions.
    void RecordReorderedUpload(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> copies,
                               u64 prior_tick);

    /// Send work to a separate thread.
    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer, vk::CommandBuffer>
    void RecordWithUploadBuffer(T&& command,
                                std::source_location location = std::source_location::current()) {
        RecordAs(std::forward<T>(command), location, RecordSite::Own, 0);
    }

    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer>
    void Record(T&& c, std::source_location location = std::source_location::current()) {
        RecordAs(
            [command = std::move(c)](vk::CommandBuffer cmdbuf, vk::CommandBuffer) {
                command(cmdbuf);
            },
            location, RecordSite::Own, 0);
    }

    /// Whether the recording thread submitted `tick` within 20 ms.
    [[nodiscard]] bool WaitForSubmission(u64 tick) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{20};
        while (tick >= master_semaphore->CurrentTick()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{100});
        }
        return true;
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore->CurrentTick();
    }

    /// Changes whenever recorded command buffer state may have been reset (InvalidateState, the
    /// end of a render pass), so a binding recorded at the same epoch is still bound.
    [[nodiscard]] u64 RecordingEpoch() const noexcept {
        return recording_epoch;
    }

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return master_semaphore->IsFree(tick);
    }

    /// Waits for the given GPU tick, optionally pacing frames. Threads other than the one that
    /// records (the async present thread) pass may_flush = false: flushing from there records
    /// into the current chunk while that thread does (corrupt command buffers, crashes). They
    /// wait for the recording thread to submit the tick instead, and give up after 20 ms (it may
    /// be waiting for them).
    void Wait(u64 tick, double target_fps = 0.0, bool may_flush = true) {
        const bool trace = Common::DkbFrameTrace::Enabled() && target_fps > 0.0;
        const auto trace_start = trace ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        if (tick > 0 && (may_flush || WaitForSubmission(tick))) {
            if (tick >= master_semaphore->CurrentTick()) {
                Flush();
            }
            master_semaphore->Wait(tick);
        }
        const auto trace_waited = trace ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        double trace_requested_ms{};
        double trace_sleep_ms{};
        double trace_slept_ms{};
        if (Settings::values.use_speed_limit.GetValue() && target_fps > 0.0) {
            auto now = std::chrono::steady_clock::now();
            if (last_target_fps != target_fps) {
                frame_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / target_fps));
                max_frame_count = static_cast<int>(0.1 * target_fps);
                last_target_fps = target_fps;
                frame_counter = 0;
                start_time = now;
            }
            frame_counter++;
            auto target_time = start_time + frame_interval * frame_counter;
            if (target_time >= now) {
                constexpr auto spin_tail = std::chrono::milliseconds(1);
                auto sleep_time = target_time - now;
                if (trace) {
                    trace_requested_ms =
                        std::chrono::duration<double, std::milli>(sleep_time).count();
                }
                if (sleep_time > spin_tail * 2) {
                    if (trace) {
                        trace_sleep_ms =
                            std::chrono::duration<double, std::milli>(sleep_time - spin_tail).count();
                    }
                    const auto sleep_start = trace ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
                    std::this_thread::sleep_for(sleep_time - spin_tail);
                    if (trace) {
                        trace_slept_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - sleep_start).count();
                    }
                }
                while (std::chrono::steady_clock::now() < target_time) {
                    std::this_thread::yield();
                }
            } else if (frame_counter > max_frame_count) {
                frame_counter = 0;
                start_time = now;
            }
        }
        if (trace) {
            static thread_local Common::DkbFrameTrace::RateLimit rate;
            const auto end = std::chrono::steady_clock::now();
            if (rate.Ready(end)) {
                LOG_INFO(Render_Vulkan,
                         "DKB host_pace target_fps={} enabled={} may_flush={} "
                         "cpu_wait_wall_ms={:.3f} pace_requested_ms={:.3f} "
                         "sleep_requested_ms={:.3f} sleep_actual_wall_ms={:.3f} "
                         "pace_actual_wall_ms={:.3f}",
                         target_fps, Settings::values.use_speed_limit.GetValue(), may_flush,
                         std::chrono::duration<double, std::milli>(trace_waited - trace_start).count(),
                         trace_requested_ms, trace_sleep_ms, trace_slept_ms,
                         std::chrono::duration<double, std::milli>(end - trace_waited).count());
            }
        }
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore& GetMasterSemaphore() const noexcept {
        return *master_semaphore;
    }

    std::mutex submit_mutex;

private:
    std::chrono::steady_clock::time_point last_flush{};
    bool flush_pending{};

    class Command {
    public:
        virtual ~Command() = default;

        virtual void Execute(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) const = 0;

        Command* GetNext() const {
            return next;
        }

        void SetNext(Command* next_) {
            next = next_;
        }

    private:
        Command* next = nullptr;
    };

    template <typename T>
    class TypedCommand final : public Command {
    public:
        explicit TypedCommand(T&& command_) : command{std::move(command_)} {}
        ~TypedCommand() override = default;

        TypedCommand(TypedCommand&&) = delete;
        TypedCommand& operator=(TypedCommand&&) = delete;

        void Execute(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) const override {

            command(cmdbuf, upload_cmdbuf);
        }

    private:
        T command;
    };

    class CommandChunk final {
    public:
        /// cost: the TEST GPU cost trace when this chunk is in a submission it samples.
        void ExecuteAll(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf, GpuCost* cost);

        /// TEST (EDEN_NX2_GPU_COST): the site of each command, in order.
        void NoteSite(const RecordSite& site) {
            sites.push_back(site);
        }

        template <typename T>
        bool Record(T& command) {
            using FuncType = TypedCommand<T>;
            static_assert(sizeof(FuncType) < sizeof(data), "Lambda is too large");
            // The offset is aligned within data, so data itself must be aligned at least as much.
            static_assert(alignof(FuncType) <= DataAlignment, "Lambda needs more alignment");

            command_offset = Common::AlignUp(command_offset, alignof(FuncType));
            if (command_offset > sizeof(data) - sizeof(FuncType)) {
                return false;
            }
            Command* const current_last = last;
            last = new (data.data() + command_offset) FuncType(std::move(command));

            if (current_last) {
                current_last->SetNext(last);
            } else {
                first = last;
            }
            command_offset += sizeof(FuncType);
            return true;
        }

        void MarkSubmit() {
            submit = true;
        }

        bool Empty() const {
            return command_offset == 0;
        }

        bool HasSubmit() const {
            return submit;
        }

    private:
        Command* first = nullptr;
        Command* last = nullptr;

        size_t command_offset = 0;
        bool submit = false;
        std::vector<RecordSite> sites;
        /// Commands are constructed in data at offsets aligned for their type. Some hold
        /// alignas(16) members (the ASTC decoder pass's BlockLinearSwizzle2DParams), and
        /// std::max_align_t is only 8 on MSVC and clang-cl, so data asks for 16 itself rather
        /// than depending on where the members before it happen to end.
        static constexpr size_t DataAlignment =
            alignof(std::max_align_t) > 16 ? alignof(std::max_align_t) : 16;
        alignas(DataAlignment) std::array<u8, 0x8000> data{};
    };

    template <typename T>
    void RecordAs(T&& command, std::source_location location, RecordSite::Role role, u8 flags) {
        if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
            VideoCommon::GpuSplit::CheckOwner("Scheduler::Record");
        }
        if (uploads_pending) {
            // Uploads recorded before this command go before it.
            FlushUploads();
        }
        if (!chunk->Record(command)) {
            DispatchWork();
            (void)chunk->Record(command);
        }
        if (record_sites) [[unlikely]] {
            chunk->NoteSite({location.file_name(), location.function_name(), location.line(), role,
                             flags});
        }
    }

    struct State {
        VkRenderPass renderpass{};
        VkFramebuffer framebuffer{};
        VkExtent2D render_area = {0, 0};
        GraphicsPipeline* graphics_pipeline = nullptr;
        bool is_rescaling = false;
        bool rescaling_defined = false;
        bool mesh_layout = false;
        bool needs_state_enable_refresh = false;
        u32 descriptor_buffer_chunk = 0;
        bool descriptor_buffer_bound = false;
    };

    struct DeferredClear {
        const Framebuffer* framebuffer = nullptr;
        u32 color_clear_mask = 0;
        std::array<VkClearValue, 8> color_values{};
        bool depth_stencil = false;
        VkClearValue depth_stencil_value{};
    };

    /// Begins a render pass for the given framebuffer, optionally with clear values.
    void BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                             const VkClearValue* clear_values, u32 clear_value_count);

    /// If a deferred clear is pending.
    void RealizeDeferredClear();

    void WorkerThread(std::stop_token stop_token);

    void AllocateWorkerCommandBuffer();

    void TraceGpuSubmission();
    void ReportWorkTimes();
    void TraceShader(u64 hash, u32 stage);

    struct ReadTraceState;
    void CreateReadTrace();
    [[nodiscard]] VideoCommon::ReadTrace::Timeline PlaceRead(
        const VideoCommon::ReadTrace::Read& read) const;

    /// cause: the line that asked for the submission, which ends a render pass still open.
    u64 SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                        std::source_location cause);

    void AllocateNewContext();

    void EndPendingOperations(std::source_location cause);

    /// cause: the line that needs the render pass ended (TEST EDEN_NX2_GPU_COST's site of it).
    void EndRenderPass(std::source_location cause = std::source_location::current());

    void AcquireNewChunk();

    const Device& device;
    StateTracker& state_tracker;

    std::unique_ptr<MasterSemaphore> master_semaphore;
    FragmentBarrierState fragment_barrier;
    bool render_pass_has_stores{};
    bool compute_writes_pending{};
    bool graphics_pipeline_missing{};
    u64 render_pass_serial{};
    std::unique_ptr<CommandPool> command_pool;

    vk::QueryPool trace_query_pool;
    static constexpr u32 TraceMaxSubmissions = 512;
    struct SubmissionTrace {
        struct ShaderUse {
            u64 hash{};
            u32 stage{};
        };
        u64 tick{};
        u32 draw_attempts{};
        u32 dispatch_attempts{};
        u32 num_shaders{};
        u32 untracked_uses{};
        std::array<ShaderUse, 32> shaders{};
    };
    SubmissionTrace trace_work;
    std::vector<SubmissionTrace> trace_submissions;
    static constexpr u32 TraceMaxWork = 4096;
    struct WorkTrace {
        u64 hash{};
        u64 fragment_hash{};
        /// Dispatch grid, or vertex and instance counts of a draw; zero when indirect.
        std::array<u32, 3> size{};
    };
    vk::QueryPool trace_work_pool;
    std::vector<WorkTrace> trace_work_items;
    u32 trace_work_count{};
    Common::DkbFrameTrace::RateLimit trace_gpu_rate;
    u64 trace_gpu_tick{};
    u64 trace_frame_end{};
    u64 trace_sample_frame{};
    u32 trace_gpu_count{};
    u64 trace_gpu_skipped{};
    u32 trace_timestamp_bits{};
    float trace_timestamp_period{};
    bool trace_gpu_recording{};
    bool trace_gpu_sampling{};
    bool trace_gpu_pending{};
    bool trace_gpu_failed{};

    /// TEST: EDEN_NX2_READ_TRACE
    std::unique_ptr<ReadTraceState> read_trace;
    VideoCommon::ReadTrace::Counts trace_counts{};

    /// TEST: EDEN_NX2_GPU_COST. record_sites: every command notes its RecordSite.
    std::unique_ptr<GpuCost> gpu_cost;
    bool record_sites{};

    /// RecordUpload's copies not recorded yet; uploads_pending while it holds any.
    std::unique_ptr<UploadBatch> upload_batch;
    std::source_location upload_batch_site{};
    bool uploads_pending{};
    /// Records the pending uploads as one command.
    void FlushUploads();
    /// RecordReorderedUpload's copies, recorded when the submission is.
    std::unique_ptr<ReorderedUploadBatch> reordered_uploads;
    bool reordered_uploads_pending{};
    /// Records the reordered uploads into the upload command buffer as one command.
    void FlushReorderedUploads();
    /// The line that ended the last render pass: the site of the next pass's begin.
    std::source_location last_pass_end{};

    VideoCommon::QueryCacheBase<QueryCacheParams>* query_cache = nullptr;

    vk::CommandBuffer current_cmdbuf;
    vk::CommandBuffer current_upload_cmdbuf;

    DeferredClear deferred_clear;

    std::unique_ptr<CommandChunk> chunk;
    std::function<void()> on_submit;

    State state;
    u64 recording_epoch = 0;

    u32 num_renderpass_images = 0;
    std::array<VkImage, 9> renderpass_images{};
    std::array<VkImageSubresourceRange, 9> renderpass_image_ranges{};

    std::queue<std::unique_ptr<CommandChunk>> work_queue;
    std::vector<std::unique_ptr<CommandChunk>> chunk_reserve;
    std::mutex execution_mutex;
    std::mutex reserve_mutex;
    std::mutex queue_mutex;
    std::condition_variable_any event_cv;
    bool worker_waiting{}; ///< The worker is waiting for work (guarded by queue_mutex)
    std::jthread worker_thread;

    std::chrono::steady_clock::duration frame_interval{};
    std::chrono::steady_clock::time_point start_time{};
    double last_target_fps{};
    u64 max_frame_count{};
    u64 frame_counter{};
};

} // namespace Vulkan
