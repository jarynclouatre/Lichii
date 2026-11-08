// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_gpu_cost.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {

enum class Op : u8 {
    SubmitStart, ///< top of pipe at the submission's start: the origin of its intervals
    PriorTail,   ///< until the work submitted before it is done
    MainStart,   ///< from the upload command buffer's end to the main one's start
    SubmitEnd,   ///< from the last command's work to the submission's end
    Draw,
    DrawIndirect,
    Dispatch,
    DispatchIndirect,
    PassBegin,
    PassBeginClear,
    PassResume, ///< a begin on the framebuffer of the pass ended last
    PassResumeClear,
    PassEnd,
    Barrier,
    ImageBarrier,
    LayoutBarrier, ///< an image barrier that changes a layout
    WaitEvents,
    SetEvent,
    CopyBuffer,
    CopyImage,
    CopyBufferToImage,
    CopyImageToBuffer,
    BlitImage,
    ResolveImage,
    ClearColorImage,
    ClearAttachments,
    FillBuffer,
    QueryBegin,
    QueryEnd,
    QueryReset,
    QueryCopy,
    ConditionalBegin,
    ConditionalEnd,
    XfbBegin,
    XfbEnd,
    Count,
};
constexpr size_t NUM_OPS = static_cast<size_t>(Op::Count);

constexpr std::array<const char*, NUM_OPS> OP_NAMES{
    "submission start", "prior work", "main start", "submission end", "draw", "indirect draw",
    "dispatch", "indirect dispatch", "begin", "begin+clear", "resume", "resume+clear", "end",
    "barrier", "image barrier", "layout change", "wait events", "set event", "copy buffer",
    "copy image", "buffer to image", "image to buffer", "blit", "resolve", "clear image",
    "clear attachments", "fill buffer", "query begin", "query end", "query reset", "query copy",
    "conditional begin", "conditional end", "xfb begin", "xfb end",
};

static_assert(OP_NAMES[NUM_OPS - 1] != nullptr);

constexpr bool IsMarker(Op op) {
    return op <= Op::SubmitEnd;
}

constexpr bool IsPassBegin(Op op) {
    return op >= Op::PassBegin && op <= Op::PassResumeClear;
}

constexpr bool IsBarrier(Op op) {
    return op >= Op::Barrier && op <= Op::SetEvent;
}

/// What ran between a render pass's end and the next begin.
enum Between : u8 {
    BetweenBarrier = 1 << 0,
    BetweenCopy = 1 << 1,
    BetweenDispatch = 1 << 2,
    BetweenQuery = 1 << 3,
    BetweenClear = 1 << 4,
    BetweenOther = 1 << 5,
};
constexpr size_t NUM_BETWEEN = 6;
constexpr std::array<const char*, NUM_BETWEEN> BETWEEN_NAMES{
    "barriers", "copies", "dispatches", "queries", "clears", "other",
};
static_assert(BETWEEN_NAMES[NUM_BETWEEN - 1] != nullptr);

constexpr u8 BetweenBit(Op op) {
    if (IsBarrier(op)) {
        return BetweenBarrier;
    }
    switch (op) {
    case Op::CopyBuffer:
    case Op::CopyImage:
    case Op::CopyBufferToImage:
    case Op::CopyImageToBuffer:
    case Op::BlitImage:
    case Op::ResolveImage:
        return BetweenCopy;
    case Op::Dispatch:
    case Op::DispatchIndirect:
        return BetweenDispatch;
    case Op::QueryBegin:
    case Op::QueryEnd:
    case Op::QueryReset:
    case Op::QueryCopy:
        return BetweenQuery;
    case Op::ClearColorImage:
    case Op::FillBuffer:
        return BetweenClear;
    default:
        return BetweenOther;
    }
}

/// The groups of the "GPU cost" line; every timed interval is in exactly one.
enum class Category : u8 {
    Draws,
    Dispatches,
    PassEnds,
    PassBegins,
    Barriers,
    Uploads,
    BufferCopies,
    ImageUploads,
    ImageDownloads,
    ImageCopies,
    Clears,
    Queries,
    Other,
    Submissions,
    Present,
    Count,
};
constexpr size_t NUM_CATEGORIES = static_cast<size_t>(Category::Count);
constexpr std::array<const char*, NUM_CATEGORIES> CATEGORY_NAMES{
    "draws",         "dispatches",    "pass ends (with the barrier after)",
    "pass begins",   "barriers",      "uploads (upload command buffer)",
    "buffer copies", "image uploads", "image downloads",
    "image copies",  "clears",        "queries",
    "conditional/xfb", "submissions", "present path",
};
static_assert(CATEGORY_NAMES[NUM_CATEGORIES - 1] != nullptr);

constexpr Category CategoryOf(RecordSite::Role role, Op op, bool upload, bool present) {
    if (present) {
        return Category::Present;
    }
    if (IsMarker(op) || role == RecordSite::Submit) {
        return Category::Submissions;
    }
    if (role == RecordSite::PassEnd || op == Op::PassEnd) {
        return Category::PassEnds;
    }
    if (role == RecordSite::PassBegin || IsPassBegin(op)) {
        return Category::PassBegins;
    }
    if (IsBarrier(op)) {
        return Category::Barriers;
    }
    switch (op) {
    case Op::Draw:
    case Op::DrawIndirect:
        return Category::Draws;
    case Op::Dispatch:
    case Op::DispatchIndirect:
        return Category::Dispatches;
    case Op::CopyBuffer:
        return upload ? Category::Uploads : Category::BufferCopies;
    case Op::CopyBufferToImage:
        return Category::ImageUploads;
    case Op::CopyImageToBuffer:
        return Category::ImageDownloads;
    case Op::CopyImage:
    case Op::BlitImage:
    case Op::ResolveImage:
        return Category::ImageCopies;
    case Op::ClearColorImage:
    case Op::ClearAttachments:
    case Op::FillBuffer:
        return Category::Clears;
    case Op::QueryBegin:
    case Op::QueryEnd:
    case Op::QueryReset:
    case Op::QueryCopy:
        return Category::Queries;
    default:
        return Category::Other;
    }
}

constexpr u32 NO_SITE = ~0u;
constexpr u32 NO_EPISODE = ~0u;
constexpr u32 CAPACITY = 32768;
/// Kept for the submissions' start and end timestamps once the others overflow.
constexpr u32 MARKER_RESERVE = 1024;

enum class Next : u8 {
    None,  ///< no begin in the sampled frame after it
    Same,  ///< the next begin has the framebuffer of the ended pass
    Other, ///< another framebuffer
};

struct TimedOp {
    u32 site;
    u32 episode;
    u32 bytes;
    u16 submission;
    Op op;
    bool upload;
};

/// One render pass end and what came before the next begin.
struct Episode {
    u32 site;
    u32 draws;
    VkFramebuffer framebuffer;
    u8 between;
    Next next;
    bool stores;
    bool crossed; ///< a submission ended before the next begin
    u64 end_ns;
    u64 between_ns;
    u64 begin_ns;
};

struct SiteInfo {
    std::string name;
    RecordSite::Role role;
    bool present;
};

struct Cell {
    u64 ns;
    u64 count;
    u64 bytes;
};

struct QueryResult {
    u64 value;
    u64 available;
};

struct BreakTotals {
    u64 count;
    u64 draws;
    u64 stores;
    u64 same;
    u64 other;
    u64 none;
    u64 crossed;
    u64 only_barriers;
    u64 nothing;
    std::array<u64, NUM_BETWEEN> between;
    u64 end_ns;
    u64 between_ns;
    u64 begin_ns;
};

std::string_view BaseName(std::string_view path) {
    if (const size_t slash = path.find_last_of("/\\"); slash != std::string_view::npos) {
        path.remove_prefix(slash + 1);
    }
    return path;
}

/// "Class::Function" from a compiler's function signature (MSVC's has the return type, the
/// calling convention and the parameters; a lambda's ends in "operator ()").
std::string ShortFunction(std::string_view signature) {
    std::string name;
    int depth = 0;
    for (const char c : signature) {
        if (c == '(' && depth == 0) {
            break;
        }
        if (c == '<') {
            ++depth;
        } else if (c == '>') {
            depth = (std::max)(depth - 1, 0);
        } else if (depth == 0) {
            name.push_back(c);
        }
    }
    const auto trim = [&name] {
        while (!name.empty() && (name.back() == ' ' || name.back() == ':')) {
            name.pop_back();
        }
    };
    trim();
    if (name.ends_with("operator")) {
        name.resize(name.size() - 8);
        trim();
    }
    if (const size_t space = name.rfind(' '); space != std::string::npos) {
        name.erase(0, space + 1);
    }
    size_t doubled = name.find("::::");
    while (doubled != std::string::npos) {
        name.erase(doubled, 2);
        doubled = name.find("::::");
    }
    if (const size_t last = name.rfind("::"); last != std::string::npos && last > 0) {
        if (const size_t before = name.rfind("::", last - 1); before != std::string::npos) {
            name.erase(0, before + 2);
        }
    }
    return name;
}

bool IsPresentPath(std::string_view path) {
    return path.find("present/") != std::string_view::npos ||
           path.find("present\\") != std::string_view::npos ||
           path.find("vk_blit_screen") != std::string_view::npos ||
           path.find("renderer_vulkan.cpp") != std::string_view::npos ||
           path.find("vk_present_manager") != std::string_view::npos;
}

} // Anonymous namespace

struct GpuCost::Impl {
    Impl(const Device& device_, vk::QueryPool pool_, u32 valid_bits_, double ns_per_tick_)
        : device{device_}, real{device_.GetDispatchLoader()}, traced{real},
          pool{std::move(pool_)}, valid_bits{valid_bits_}, ns_per_tick{ns_per_tick_},
          host_reset{device_.IsHostQueryResetSupported()} {
        stamps.resize(CAPACITY);
        window_start = std::chrono::steady_clock::now();
    }

    void Stamp(VkCommandBuffer cmdbuf, Op op, u64 bytes, u32 episode) {
        if (!open) {
            return;
        }
        if (count >= (IsMarker(op) ? CAPACITY : CAPACITY - MARKER_RESERVE)) {
            ++frame_overflow;
            return;
        }
        real.vkCmdWriteTimestamp(cmdbuf,
                                 op == Op::SubmitStart ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                       : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 *pool, count);
        stamps[count] = {
            .site = current_site,
            .episode = episode,
            .bytes = static_cast<u32>((std::min)(bytes, u64{0xffffffff})),
            .submission = submission,
            .op = op,
            .upload = cmdbuf == upload_handle,
        };
        ++count;
    }

    /// Any command that does work, after it is recorded.
    void Work(VkCommandBuffer cmdbuf, Op op, u64 bytes) {
        if (cmdbuf == upload_handle) {
            Stamp(cmdbuf, op, bytes, NO_EPISODE);
            return;
        }
        if ((op == Op::Draw || op == Op::DrawIndirect) && in_pass) {
            ++pass_draws;
        }
        if (pending != NO_EPISODE && current_role == RecordSite::Own) {
            Episode& episode = episodes[pending];
            episode.between = static_cast<u8>(episode.between | BetweenBit(op));
        }
        Stamp(cmdbuf, op, bytes, pending);
    }

    void PassEnded(VkCommandBuffer cmdbuf) {
        if (open) {
            pending = static_cast<u32>(episodes.size());
            episodes.push_back({
                .site = current_site,
                .draws = pass_draws,
                .framebuffer = pass_framebuffer,
                .between = 0,
                .next = Next::None,
                .stores = (current_flags & RecordSite::PassHadStores) != 0,
                .crossed = false,
                .end_ns = 0,
                .between_ns = 0,
                .begin_ns = 0,
            });
        }
        in_pass = false;
        Stamp(cmdbuf, Op::PassEnd, 0, pending);
    }

    void PassBegan(VkCommandBuffer cmdbuf, const VkRenderPassBeginInfo& info) {
        const u32 closing = pending;
        bool same = false;
        if (closing != NO_EPISODE) {
            Episode& episode = episodes[closing];
            same = episode.framebuffer == info.framebuffer;
            episode.next = same ? Next::Same : Next::Other;
            pending = NO_EPISODE;
        }
        in_pass = true;
        pass_framebuffer = info.framebuffer;
        pass_draws = 0;
        const bool clear = info.clearValueCount != 0;
        const Op op = same ? (clear ? Op::PassResumeClear : Op::PassResume)
                           : (clear ? Op::PassBeginClear : Op::PassBegin);
        Stamp(cmdbuf, op, 0, closing);
    }

    u32 Intern(const RecordSite& site) {
        const auto key = std::make_pair(std::make_pair(site.file, site.line),
                                        static_cast<u32>(site.role));
        if (const auto it = site_keys.find(key); it != site_keys.end()) {
            return it->second;
        }
        std::string name = SiteName(site.file, site.line, site.function);
        const auto [named, inserted] =
            site_names.try_emplace(fmt::format("{}#{}", name, static_cast<u32>(site.role)),
                                   static_cast<u32>(sites.size()));
        if (inserted) {
            sites.push_back({
                .name = std::move(name),
                .role = site.role,
                .present = IsPresentPath(site.file),
            });
        }
        site_keys.emplace(key, named->second);
        return named->second;
    }

    void StartFrame() {
        count = 0;
        submission = 0;
        frame_overflow = 0;
        episodes.clear();
        pending = NO_EPISODE;
        in_pass = false;
        frame_open = true;
    }

    [[nodiscard]] u64 DeltaNs(u64 begin, u64 end) const {
        const u64 ticks = Common::DkbFrameTrace::TimestampDelta(begin, end, valid_bits);
        return static_cast<u64>(static_cast<double>(ticks) * ns_per_tick);
    }

    void Attribute(u32 index, u64 ns) {
        const TimedOp& stamp = stamps[index];
        const bool has_site = stamp.site != NO_SITE;
        const RecordSite::Role role = has_site ? sites[stamp.site].role : RecordSite::Own;
        const bool present = has_site && sites[stamp.site].present;
        Cell& cell = cells[static_cast<size_t>(role)][static_cast<size_t>(stamp.op)]
                          [stamp.upload ? 1 : 0][present ? 1 : 0];
        cell.ns += ns;
        ++cell.count;
        cell.bytes += stamp.bytes;
        if (has_site) {
            Cell& site_cell = by_site[(u64{stamp.site} << 8) | static_cast<u64>(stamp.op)];
            site_cell.ns += ns;
            ++site_cell.count;
            site_cell.bytes += stamp.bytes;
        }
        if (stamp.episode == NO_EPISODE) {
            return;
        }
        Episode& episode = episodes[stamp.episode];
        if (stamp.op == Op::PassEnd || role == RecordSite::PassEnd) {
            episode.end_ns += ns;
        } else if (IsPassBegin(stamp.op) || role == RecordSite::PassBegin) {
            episode.begin_ns += ns;
        } else {
            episode.between_ns += ns;
        }
    }

    void ReadFrame();
    void Report();

    const Device& device;
    const vk::DeviceDispatch real;
    vk::DeviceDispatch traced;
    vk::QueryPool pool;
    const u32 valid_bits;
    const double ns_per_tick;
    const bool host_reset;

    // The sampled frame (worker thread).
    std::vector<TimedOp> stamps;
    std::vector<Episode> episodes;
    u32 count{};
    u16 submission{};
    u64 frame_overflow{};
    bool frame_open{};
    bool open{}; ///< in a sampled submission
    VkCommandBuffer main_handle{};
    VkCommandBuffer upload_handle{};

    // The command running.
    u32 current_site = NO_SITE;
    RecordSite::Role current_role = RecordSite::Own;
    u8 current_flags{};

    // The render pass state the commands left.
    bool in_pass{};
    VkFramebuffer pass_framebuffer{};
    u32 pass_draws{};
    u32 pending = NO_EPISODE; ///< the episode of the last end, until the next begin

    // Source lines seen, by pointer and by name (a header's line has a pointer in each TU).
    std::vector<SiteInfo> sites;
    std::unordered_map<std::string, u32> site_names;
    struct KeyHash {
        size_t operator()(const std::pair<std::pair<const char*, u32>, u32>& key) const {
            return std::hash<const char*>{}(key.first.first) ^
                   (static_cast<size_t>(key.first.second) << 8) ^ key.second;
        }
    };
    std::unordered_map<std::pair<std::pair<const char*, u32>, u32>, u32, KeyHash> site_keys;

    // The 10 s window.
    std::chrono::steady_clock::time_point window_start;
    std::array<std::array<std::array<std::array<Cell, 2>, 2>, NUM_OPS>, 4> cells{};
    std::unordered_map<u64, Cell> by_site;
    std::unordered_map<u32, BreakTotals> breaks;
    u64 frames{};
    u64 busy_ns{};
    u64 submissions{};
    u64 stamped{};
    u64 overflow{};
    u64 bad{};
    std::vector<QueryResult> results;
};

namespace {

GpuCost::Impl* g_impl{};

/// A vkCmd* entry that runs the real one, then notes what it did.
template <typename Pfn>
struct Hook;

template <typename... Args>
struct Hook<void(VKAPI_PTR*)(VkCommandBuffer, Args...)> {
    template <auto member, Op op>
    static VKAPI_ATTR void VKAPI_CALL Call(VkCommandBuffer cmdbuf, Args... args) {
        GpuCost::Impl& self = *g_impl;
        (self.real.*member)(cmdbuf, args...);
        self.Work(cmdbuf, op, 0);
    }
};

template <auto member, Op op>
void Install(vk::DeviceDispatch& traced) {
    using Pfn = std::remove_reference_t<decltype(traced.*member)>;
    if (traced.*member != nullptr) {
        traced.*member = &Hook<Pfn>::template Call<member, op>;
    }
}

Op BarrierOp(u32 image_count, const VkImageMemoryBarrier* images) {
    if (image_count == 0) {
        return Op::Barrier;
    }
    for (u32 i = 0; i < image_count; ++i) {
        if (images[i].oldLayout != images[i].newLayout) {
            return Op::LayoutBarrier;
        }
    }
    return Op::ImageBarrier;
}

VKAPI_ATTR void VKAPI_CALL PipelineBarrier(VkCommandBuffer cmdbuf, VkPipelineStageFlags src,
                                           VkPipelineStageFlags dst, VkDependencyFlags flags,
                                           u32 memory_count, const VkMemoryBarrier* memory,
                                           u32 buffer_count, const VkBufferMemoryBarrier* buffers,
                                           u32 image_count, const VkImageMemoryBarrier* images) {
    GpuCost::Impl& self = *g_impl;
    self.real.vkCmdPipelineBarrier(cmdbuf, src, dst, flags, memory_count, memory, buffer_count,
                                   buffers, image_count, images);
    self.Work(cmdbuf, BarrierOp(image_count, images), 0);
}

VKAPI_ATTR void VKAPI_CALL PipelineBarrier2(VkCommandBuffer cmdbuf, const VkDependencyInfo* info) {
    GpuCost::Impl& self = *g_impl;
    self.real.vkCmdPipelineBarrier2(cmdbuf, info);
    Op op = info->imageMemoryBarrierCount == 0 ? Op::Barrier : Op::ImageBarrier;
    for (u32 i = 0; i < info->imageMemoryBarrierCount; ++i) {
        if (info->pImageMemoryBarriers[i].oldLayout != info->pImageMemoryBarriers[i].newLayout) {
            op = Op::LayoutBarrier;
        }
    }
    self.Work(cmdbuf, op, 0);
}

VKAPI_ATTR void VKAPI_CALL CopyBuffer(VkCommandBuffer cmdbuf, VkBuffer src, VkBuffer dst,
                                      u32 region_count, const VkBufferCopy* regions) {
    GpuCost::Impl& self = *g_impl;
    self.real.vkCmdCopyBuffer(cmdbuf, src, dst, region_count, regions);
    u64 bytes = 0;
    for (u32 i = 0; i < region_count; ++i) {
        bytes += regions[i].size;
    }
    self.Work(cmdbuf, Op::CopyBuffer, bytes);
}

VKAPI_ATTR void VKAPI_CALL BeginRenderPass(VkCommandBuffer cmdbuf,
                                           const VkRenderPassBeginInfo* info,
                                           VkSubpassContents contents) {
    GpuCost::Impl& self = *g_impl;
    self.real.vkCmdBeginRenderPass(cmdbuf, info, contents);
    self.PassBegan(cmdbuf, *info);
}

VKAPI_ATTR void VKAPI_CALL EndRenderPass(VkCommandBuffer cmdbuf) {
    GpuCost::Impl& self = *g_impl;
    self.real.vkCmdEndRenderPass(cmdbuf);
    self.PassEnded(cmdbuf);
}

void InstallHooks(vk::DeviceDispatch& traced) {
    using D = vk::DeviceDispatch;
    Install<&D::vkCmdDraw, Op::Draw>(traced);
    Install<&D::vkCmdDrawIndexed, Op::Draw>(traced);
    Install<&D::vkCmdDrawMeshTasksEXT, Op::Draw>(traced);
    Install<&D::vkCmdDrawIndirect, Op::DrawIndirect>(traced);
    Install<&D::vkCmdDrawIndexedIndirect, Op::DrawIndirect>(traced);
    Install<&D::vkCmdDrawIndirectCount, Op::DrawIndirect>(traced);
    Install<&D::vkCmdDrawIndexedIndirectCount, Op::DrawIndirect>(traced);
    Install<&D::vkCmdDrawIndirectByteCountEXT, Op::DrawIndirect>(traced);
    Install<&D::vkCmdDispatch, Op::Dispatch>(traced);
    Install<&D::vkCmdDispatchIndirect, Op::DispatchIndirect>(traced);
    Install<&D::vkCmdWaitEvents, Op::WaitEvents>(traced);
    Install<&D::vkCmdSetEvent, Op::SetEvent>(traced);
    Install<&D::vkCmdCopyImage, Op::CopyImage>(traced);
    Install<&D::vkCmdCopyBufferToImage, Op::CopyBufferToImage>(traced);
    Install<&D::vkCmdCopyImageToBuffer, Op::CopyImageToBuffer>(traced);
    Install<&D::vkCmdBlitImage, Op::BlitImage>(traced);
    Install<&D::vkCmdResolveImage, Op::ResolveImage>(traced);
    Install<&D::vkCmdClearColorImage, Op::ClearColorImage>(traced);
    Install<&D::vkCmdClearAttachments, Op::ClearAttachments>(traced);
    Install<&D::vkCmdFillBuffer, Op::FillBuffer>(traced);
    Install<&D::vkCmdBeginQuery, Op::QueryBegin>(traced);
    Install<&D::vkCmdEndQuery, Op::QueryEnd>(traced);
    Install<&D::vkCmdResetQueryPool, Op::QueryReset>(traced);
    Install<&D::vkCmdCopyQueryPoolResults, Op::QueryCopy>(traced);
    Install<&D::vkCmdBeginConditionalRenderingEXT, Op::ConditionalBegin>(traced);
    Install<&D::vkCmdEndConditionalRenderingEXT, Op::ConditionalEnd>(traced);
    Install<&D::vkCmdBeginTransformFeedbackEXT, Op::XfbBegin>(traced);
    Install<&D::vkCmdEndTransformFeedbackEXT, Op::XfbEnd>(traced);
    if (traced.vkCmdPipelineBarrier) {
        traced.vkCmdPipelineBarrier = &PipelineBarrier;
    }
    if (traced.vkCmdPipelineBarrier2) {
        traced.vkCmdPipelineBarrier2 = &PipelineBarrier2;
    }
    if (traced.vkCmdCopyBuffer) {
        traced.vkCmdCopyBuffer = &CopyBuffer;
    }
    if (traced.vkCmdBeginRenderPass) {
        traced.vkCmdBeginRenderPass = &BeginRenderPass;
    }
    if (traced.vkCmdEndRenderPass) {
        traced.vkCmdEndRenderPass = &EndRenderPass;
    }
}

} // Anonymous namespace

void GpuCost::Impl::ReadFrame() {
    results.assign(count, {});
    const VkResult result = device.GetLogical().GetQueryResults(
        *pool, 0, count, results.size() * sizeof(results[0]), results.data(), sizeof(results[0]),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS) {
        ++bad;
        return;
    }
    std::vector<std::vector<u32>> upload_order(submission);
    std::vector<std::vector<u32>> main_order(submission);
    for (u32 i = 0; i < count; ++i) {
        const TimedOp& stamp = stamps[i];
        if (stamp.submission < submission) {
            (stamp.upload ? upload_order : main_order)[stamp.submission].push_back(i);
        }
    }
    for (u16 s = 0; s < submission; ++s) {
        std::vector<u32>& order = upload_order[s];
        order.insert(order.end(), main_order[s].begin(), main_order[s].end());
        if (order.empty() || stamps[order.front()].op != Op::SubmitStart ||
            stamps[order.back()].op != Op::SubmitEnd ||
            std::ranges::any_of(order, [&](u32 i) { return results[i].available == 0; })) {
            ++bad;
            continue;
        }
        u64 previous = results[order.front()].value;
        u64 total = 0;
        for (size_t k = 1; k < order.size(); ++k) {
            const u64 value = results[order[k]].value;
            u64 ns = DeltaNs(previous, value);
            if (ns > u64{1'000'000'000}) {
                // Out of order: the interval is lost, the rest stays right.
                ++bad;
                ns = 0;
            }
            previous = value;
            total += ns;
            Attribute(order[k], ns);
        }
        busy_ns += total;
    }
    for (const Episode& episode : episodes) {
        BreakTotals& totals = breaks[episode.site];
        ++totals.count;
        totals.draws += episode.draws;
        const auto one = [](bool condition) { return condition ? u64{1} : u64{0}; };
        totals.stores += one(episode.stores);
        totals.same += one(episode.next == Next::Same);
        totals.other += one(episode.next == Next::Other);
        totals.none += one(episode.next == Next::None);
        totals.crossed += one(episode.crossed);
        totals.only_barriers += one(episode.between == BetweenBarrier);
        totals.nothing += one(episode.between == 0);
        for (size_t bit = 0; bit < NUM_BETWEEN; ++bit) {
            totals.between[bit] += one(((episode.between >> bit) & 1u) != 0);
        }
        totals.end_ns += episode.end_ns;
        totals.between_ns += episode.between_ns;
        totals.begin_ns += episode.begin_ns;
    }
    ++frames;
    submissions += submission;
    stamped += count;
    overflow += frame_overflow;
}

void GpuCost::Impl::Report() {
    if (frames == 0) {
        return;
    }
    const double per_frame = 1.0 / static_cast<double>(frames);
    const auto ms = [per_frame](u64 ns) { return static_cast<double>(ns) / 1e6 * per_frame; };
    const auto times = [per_frame](u64 n) { return static_cast<double>(n) * per_frame; };

    // Totals by category, each with its kinds by time.
    struct Group {
        Cell total{};
        std::array<Cell, NUM_OPS> ops{};
    };
    std::array<Group, NUM_CATEGORIES> groups{};
    for (size_t role = 0; role < cells.size(); ++role) {
        for (size_t op = 0; op < NUM_OPS; ++op) {
            for (size_t upload = 0; upload < 2; ++upload) {
                for (size_t present = 0; present < 2; ++present) {
                    const Cell& cell = cells[role][op][upload][present];
                    if (cell.count == 0) {
                        continue;
                    }
                    Group& group = groups[static_cast<size_t>(
                        CategoryOf(static_cast<RecordSite::Role>(role), static_cast<Op>(op),
                                   upload != 0, present != 0))];
                    group.total.ns += cell.ns;
                    group.total.count += cell.count;
                    group.total.bytes += cell.bytes;
                    group.ops[op].ns += cell.ns;
                    group.ops[op].count += cell.count;
                }
            }
        }
    }
    std::string line;
    for (size_t c = 0; c < NUM_CATEGORIES; ++c) {
        const Group& group = groups[c];
        if (group.total.count == 0) {
            continue;
        }
        line += fmt::format(" | {} {:.3f} ms {:.1f}x", CATEGORY_NAMES[c], ms(group.total.ns),
                            times(group.total.count));
        if (group.total.bytes != 0) {
            line += fmt::format(" {:.0f} KiB", static_cast<double>(group.total.bytes) / 1024.0 *
                                                   per_frame);
        }
        std::vector<size_t> kinds;
        for (size_t op = 0; op < NUM_OPS; ++op) {
            if (group.ops[op].count != 0) {
                kinds.push_back(op);
            }
        }
        if (kinds.size() > 1) {
            std::ranges::sort(kinds, std::greater<>{},
                              [&group](size_t op) { return group.ops[op].ns; });
            line += " (";
            for (size_t k = 0; k < kinds.size(); ++k) {
                line += fmt::format("{}{} {:.3f}/{:.1f}x", k == 0 ? "" : ", ", OP_NAMES[kinds[k]],
                                    ms(group.ops[kinds[k]].ns), times(group.ops[kinds[k]].count));
            }
            line += ")";
        }
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                       window_start)
                             .count();
    LOG_INFO(Render_Vulkan,
             "GPU cost in {:.1f} s, {} sampled frames, per frame: busy {:.3f} ms in {:.1f} "
             "submissions, {:.0f} timestamps ({} over, {} bad){}",
             seconds, frames, ms(busy_ns), times(submissions), times(stamped), overflow, bad, line);

    // By source line and kind.
    std::unordered_map<std::string, Cell> named;
    for (const auto& [key, cell] : by_site) {
        const SiteInfo& site = sites[static_cast<size_t>(key >> 8)];
        const char* role = site.role == RecordSite::PassEnd     ? "[pass end] "
                           : site.role == RecordSite::PassBegin ? "[next pass] "
                           : site.role == RecordSite::Submit    ? "[submission] "
                                                                : "";
        Cell& total = named[fmt::format("{}{} at {}", role, OP_NAMES[key & 0xff], site.name)];
        total.ns += cell.ns;
        total.count += cell.count;
        total.bytes += cell.bytes;
    }
    std::vector<std::pair<std::string, Cell>> sorted(named.begin(), named.end());
    std::ranges::sort(sorted, std::greater<>{}, [](const auto& entry) { return entry.second.ns; });
    if (sorted.size() > 28) {
        sorted.resize(28);
    }
    std::string site_line;
    for (const auto& [name, cell] : sorted) {
        site_line += fmt::format("{}{:.3f} ms {:.1f}x", site_line.empty() ? "" : " | ",
                                 ms(cell.ns), times(cell.count));
        if (cell.bytes != 0) {
            site_line += fmt::format(" {:.0f} KiB",
                                     static_cast<double>(cell.bytes) / 1024.0 * per_frame);
        }
        site_line += fmt::format(" {}", name);
    }
    LOG_INFO(Render_Vulkan, "GPU cost by site, per frame: {}", site_line);

    // Render pass breaks by the line that ended the pass.
    std::unordered_map<std::string, BreakTotals> by_name;
    for (const auto& [site, totals] : breaks) {
        BreakTotals& merged = by_name[site == NO_SITE ? std::string{"unknown"} : sites[site].name];
        merged.count += totals.count;
        merged.draws += totals.draws;
        merged.stores += totals.stores;
        merged.same += totals.same;
        merged.other += totals.other;
        merged.none += totals.none;
        merged.crossed += totals.crossed;
        merged.only_barriers += totals.only_barriers;
        merged.nothing += totals.nothing;
        for (size_t bit = 0; bit < NUM_BETWEEN; ++bit) {
            merged.between[bit] += totals.between[bit];
        }
        merged.end_ns += totals.end_ns;
        merged.between_ns += totals.between_ns;
        merged.begin_ns += totals.begin_ns;
    }
    std::vector<std::pair<std::string, BreakTotals>> ends(by_name.begin(), by_name.end());
    std::ranges::sort(ends, std::greater<>{}, [](const auto& entry) { return entry.second.count; });
    if (ends.size() > 14) {
        ends.resize(14);
    }
    for (const auto& [name, totals] : ends) {
        const double n = static_cast<double>(totals.count);
        const auto percent = [n](u64 part) { return static_cast<double>(part) * 100.0 / n; };
        std::string between;
        for (size_t bit = 0; bit < NUM_BETWEEN; ++bit) {
            between +=
                fmt::format(", {} {:.0f}%", BETWEEN_NAMES[bit], percent(totals.between[bit]));
        }
        LOG_INFO(Render_Vulkan,
                 "Render pass breaks at {}: {:.1f} a frame, {:.1f} draws in each pass, storage "
                 "writes in {:.0f}%; next begin: same framebuffer {:.0f}%, another {:.0f}%, none "
                 "in the frame {:.0f}%, after a submission {:.0f}%; between: only barriers "
                 "{:.0f}%, nothing {:.0f}%{}; GPU us per break: end {:.1f}, between {:.1f}, next "
                 "begin {:.1f}",
                 name, times(totals.count), static_cast<double>(totals.draws) / n,
                 percent(totals.stores), percent(totals.same), percent(totals.other),
                 percent(totals.none), percent(totals.crossed), percent(totals.only_barriers),
                 percent(totals.nothing), between, static_cast<double>(totals.end_ns) / n / 1e3,
                 static_cast<double>(totals.between_ns) / n / 1e3,
                 static_cast<double>(totals.begin_ns) / n / 1e3);
    }

    cells = {};
    by_site.clear();
    breaks.clear();
    frames = 0;
    busy_ns = 0;
    submissions = 0;
    stamped = 0;
    overflow = 0;
    bad = 0;
    window_start = std::chrono::steady_clock::now();
}

bool GpuCost::Enabled() {
    const char* const value = std::getenv("EDEN_NX2_GPU_COST");
    return value != nullptr && *value != '\0' && *value != '0';
}

bool GpuCost::TracesUploads() {
    const char* const value = std::getenv("EDEN_NX2_GPU_COST");
    return value != nullptr && std::atoi(value) >= 2;
}

std::unique_ptr<GpuCost> GpuCost::Create(const Device& device, u32 timestamp_bits,
                                         float timestamp_period, bool frames_sampled) {
    if (!Enabled()) {
        return nullptr;
    }
    if (!frames_sampled) {
        LOG_WARNING(Render_Vulkan, "GPU cost (EDEN_NX2_GPU_COST) needs EDEN_DKB_FRAME_TRACE=1 and "
                                   "device timestamps: off");
        return nullptr;
    }
    vk::QueryPool pool;
    try {
        pool = device.GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = CAPACITY,
            .pipelineStatistics = 0,
        });
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "GPU cost: no query pool: {}", error.what());
        return nullptr;
    }
    auto impl = std::make_unique<Impl>(device, std::move(pool), timestamp_bits,
                                       static_cast<double>(timestamp_period));
    InstallHooks(impl->traced);
    if (impl->host_reset) {
        device.GetLogical().ResetQueryPool(*impl->pool, 0, CAPACITY);
    }
    g_impl = impl.get();
    LOG_INFO(Render_Vulkan,
             "GPU cost (EDEN_NX2_GPU_COST): timing every Vulkan command that does work in the "
             "frames EDEN_DKB_FRAME_TRACE samples, {} queries, {} reset",
             CAPACITY, impl->host_reset ? "host" : "command buffer");
    return std::unique_ptr<GpuCost>(new GpuCost(std::move(impl)));
}

GpuCost::GpuCost(std::unique_ptr<Impl> impl_) : impl{std::move(impl_)} {}

GpuCost::~GpuCost() {
    if (g_impl == impl.get()) {
        g_impl = nullptr;
    }
}

std::string GpuCost::SiteName(const char* file, u32 line, const char* function) {
    return fmt::format("{}:{} {}", BaseName(file), line, ShortFunction(function));
}

void GpuCost::BeginSubmission(vk::CommandBuffer& cmdbuf, vk::CommandBuffer& upload_cmdbuf,
                              bool first_of_frame) {
    Impl& self = *impl;
    if (first_of_frame) {
        self.StartFrame();
    }
    if (!self.frame_open || self.submission == 0xffff) {
        return;
    }
    self.main_handle = *cmdbuf;
    self.upload_handle = *upload_cmdbuf;
    if (first_of_frame && !self.host_reset) {
        self.real.vkCmdResetQueryPool(self.upload_handle, *self.pool, 0, CAPACITY);
    }
    self.open = true;
    self.current_site = NO_SITE;
    self.current_role = RecordSite::Own;
    self.current_flags = 0;
    self.Stamp(self.upload_handle, Op::SubmitStart, 0, NO_EPISODE);
    self.Stamp(self.upload_handle, Op::PriorTail, 0, NO_EPISODE);
    self.Stamp(self.main_handle, Op::MainStart, 0, NO_EPISODE);
    cmdbuf = vk::CommandBuffer(self.main_handle, self.traced);
    upload_cmdbuf = vk::CommandBuffer(self.upload_handle, self.traced);
}

void GpuCost::BeginCommand(const RecordSite& site) {
    Impl& self = *impl;
    self.current_site = self.Intern(site);
    self.current_role = site.role;
    self.current_flags = site.flags;
}

void GpuCost::EndSubmission(vk::CommandBuffer cmdbuf) {
    Impl& self = *impl;
    if (!self.open) {
        return;
    }
    self.Stamp(*cmdbuf, Op::SubmitEnd, 0, NO_EPISODE);
    if (self.pending != NO_EPISODE) {
        self.episodes[self.pending].crossed = true;
    }
    self.open = false;
    ++self.submission;
    self.upload_handle = VK_NULL_HANDLE;
    self.main_handle = VK_NULL_HANDLE;
}

void GpuCost::FrameDone() {
    Impl& self = *impl;
    if (!self.frame_open) {
        return;
    }
    self.frame_open = false;
    self.open = false;
    if (self.count != 0) {
        self.ReadFrame();
        if (self.host_reset) {
            self.device.GetLogical().ResetQueryPool(*self.pool, 0, CAPACITY);
        }
    }
    if (std::chrono::steady_clock::now() - self.window_start >= std::chrono::seconds{10}) {
        self.Report();
    }
}

} // namespace Vulkan
