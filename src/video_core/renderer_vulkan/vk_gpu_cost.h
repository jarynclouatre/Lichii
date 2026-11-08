// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <string>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

// TEST (EDEN_NX2_GPU_COST=1, in the frames EDEN_DKB_FRAME_TRACE=1 samples): where the host GPU's
// time goes, the part outside the timed draws and dispatches included.
//
// A sampled submission's command buffers record through a copy of the dispatch table whose
// vkCmd* entries count each call and, after each one that does work (not binds or dynamic
// state), write a bottom-of-pipe timestamp. Consecutive timestamps of a command buffer split the
// submission's GPU time, from a top-of-pipe one at its start to one at its end, into intervals
// that each end with one Vulkan command: the time from the end of the work before it to the end
// of that command's own work. Every interval goes to the command's kind (draw, render pass
// begin or end, barrier, copy, clear, query...) and to the source line that recorded it
// (Scheduler::Record's caller; a render pass's end to the line that ended it, its begin to the
// line that ended the pass before).
//
// Render pass breaks: after each end, what ran before the next begin (only barriers, copies,
// dispatches, queries, clears) and whether that begin resumed the same framebuffer.
//
// This only reads: the commands, their order and their synchronization are unchanged. Logged
// every 10 s over the frames sampled in it, per frame: "GPU cost" (totals by kind), "GPU cost
// by site", and one "Render pass breaks" line for each of the lines that ended the most passes.
// EDEN_NX2_GPU_COST=2 also logs "Uploads": where each upload's pages were last used.
namespace Vulkan {

class Device;
struct RecordSite;

class GpuCost {
public:
    /// Whether EDEN_NX2_GPU_COST is set.
    [[nodiscard]] static bool Enabled();

    /// EDEN_NX2_GPU_COST=2: also classify the uploads that run in order (vk_buffer_cache.cpp).
    /// Its page ticks cost the GPU thread several microseconds per bind of a large buffer.
    [[nodiscard]] static bool TracesUploads();

    /// Null unless EDEN_NX2_GPU_COST is set, frames are sampled (EDEN_DKB_FRAME_TRACE's
    /// timestamp pools exist) and the query pool can be created.
    static std::unique_ptr<GpuCost> Create(const Device& device, u32 timestamp_bits,
                                           float timestamp_period, bool frames_sampled);
    ~GpuCost();

    GpuCost(const GpuCost&) = delete;
    GpuCost& operator=(const GpuCost&) = delete;

    /// "file:line Class::Function" of a source location.
    [[nodiscard]] static std::string SiteName(const char* file, u32 line, const char* function);

    /// Worker thread: a sampled submission starts in these command buffers. Writes its start
    /// timestamps and switches both to the tracing dispatch table.
    void BeginSubmission(vk::CommandBuffer& cmdbuf, vk::CommandBuffer& upload_cmdbuf,
                         bool first_of_frame);
    /// Worker thread: the site of the command about to run.
    void BeginCommand(const RecordSite& site);
    /// Worker thread, in the submission's last command before its command buffers end.
    void EndSubmission(vk::CommandBuffer cmdbuf);
    /// Worker thread: every submission of the sampled frame is done on the GPU.
    void FrameDone();

    struct Impl;

private:
    explicit GpuCost(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
