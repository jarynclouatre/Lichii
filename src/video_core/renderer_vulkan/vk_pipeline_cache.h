// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stop_token>
#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_prebuild.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;

using VideoCommon::ShaderInfo;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

struct ComputePipelineCacheEntry {
    std::unique_ptr<ComputePipeline> pipeline;
    std::string failure;
    /// Constant buffer slots that were null pointers at translation (see NullStoragePointerSlots)
    std::vector<std::pair<u32, u32>> null_pointer_slots;
    /// Constant buffer slots bound as storage buffer pointers (see MappedStoragePointerKeys)
    std::vector<u64> mapped_pointers;
    u32 rebuilds{};

    [[nodiscard]] ComputePipeline* Get(bool native, GPUVAddr address,
                                       const ComputePipelineCacheKey& key) const;
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           DescriptorBufferRing& descriptor_buffer_ring,
                           RenderPassCache& render_pass_cache, BufferCache& buffer_cache,
                           TextureCache& texture_cache, VideoCore::ShaderNotify& shader_notify_);
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    void TickFrame();

    /// EDEN_NX2_HEAVY_SKIP: a clear of these targets (0 = none) in the current frame.
    void NoteClear(u64 color_address, u64 depth_address);

    [[nodiscard]] ComputePipeline* CurrentComputePipeline(u64& native_key);

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    /// What a graphics warm-up entry came to: built, not buildable (a guess the translation
    /// rejects, which another warm-up would reject too), or refused by the driver.
    enum class WarmResult { Built, Unbuildable, DriverFailed };

    /// Graphics pipeline libraries (EDEN_NX2_GPL): builds the pre-rasterization and fragment
    /// shader libraries of the pipeline `key` would make, on the calling thread, with no draw and
    /// no pipeline kept (for a warm-up). `pools` must be the calling thread's own. Unbuildable
    /// when the device has no libraries or the pipeline takes the monolithic route, DriverFailed
    /// when a library build failed.
    /// The key may hold a single stage (no partner): a vertex-only key builds just that vertex
    /// program's pre-rasterization library (vertex SPIR-V does not depend on the fragment
    /// stage); a fragment-only key builds just the fragment library, emitted as if the partner
    /// writes every varying (the common variant). The key's state (attribute types, render
    /// target formats, depth and raster state) is what the libraries are keyed and built with.
    WarmResult WarmGraphicsLibraries(ShaderPools& pools, const GraphicsPipelineCacheKey& key,
                                     std::span<Shader::Environment* const> envs);

private:
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    /// Returns `pipeline` when it is built or its draw must wait for it, nullptr when the draw
    /// may be skipped while it builds (asynchronous shaders, CanDiscardPendingDraw, heavy skip).
    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline);

    /// EDEN_NX2_HEAVY_SKIP: the current draw's render target was also drawn to in one of the two
    /// frames before this one, so it is redrawn every frame and a skipped draw only shows for a
    /// moment. A target drawn once (a baked texture) would keep the missing draw.
    [[nodiscard]] bool CurrentTargetIsRedrawn();

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools = nullptr,
        std::unique_ptr<EmissionBudget::Reservation> reservation = nullptr,
        bool libraries_only = false);

    [[nodiscard]] bool StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const;

    /// Translates the current compute program, binding the pointers in keep_pointers whatever
    /// their slots hold now (see GenericEnvironment::KeepStoragePointers)
    ComputePipelineCacheEntry CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                    const ShaderInfo* shader,
                                                    std::span<const u64> keep_pointers = {});

    ComputePipelineCacheEntry CreateComputePipeline(ShaderPools& pools,
                                                     const ComputePipelineCacheKey& key,
                                                     Shader::Environment& env,
                                                     PipelineStatistics* statistics,
                                                     bool build_in_parallel);

    void SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    void QueueVulkanPipelineCacheFlush();

    /// Queues one write of the driver pipeline cache file on the serialization thread, with no
    /// throttle. For the end of a precache: its builds never reach the timed flush, which counts
    /// only pipelines built during play.
    void QueueVulkanPipelineCacheSave();

    /// One build of the shader precache: builds a pipeline and destroys it, true when it built.
    struct PrecacheJob {
        u64 graphics_hash{}; ///< The cached graphics pipeline it builds, 0 for a compute program
        Common::UniqueFunction<bool> build;
    };

    /// Shader precache before the game starts (precache_shaders), on every CPU thread unless
    /// shader_warm_threads or EDEN_NX2_WARM_THREADS say otherwise: the cached
    /// graphics pipelines this boot skipped (`jobs`; `precached` holds those built by earlier
    /// precaches) and the compute programs of the title's precache file (made from the game files,
    /// missing reads guessed) are built, then thrown away. They only fill the driver's shader
    /// cache, so a later build of the same pipeline finds the driver's work done. Reports progress
    /// through `callback` and returns when it is done, skipped from the loading screen or the load
    /// is stopped. The graphics programs of warm-graphics.bin come last (PrecacheGraphicsWarm).
    void PrecacheShaders(u64 title_id, std::vector<PrecacheJob> jobs,
                         std::unordered_set<u64> precached, std::stop_token stop_loading,
                         const VideoCore::DiskResourceLoadCallback& callback);

    /// The graphics part of the shader precache, after PrecacheShaders' builds: the vertex and
    /// fragment programs of the title's warm-graphics.bin (made from the game files, missing
    /// reads guessed; needs graphics pipeline libraries) are built as single-stage libraries on
    /// `num_threads` threads, then released, unless a matching marker says they were built. The
    /// driver's cache keeps them, so a later pipeline with those stages costs a link. The file is
    /// streamed (it can be 1.5 GB). Reports progress through `callback` after `done_before` of
    /// `total_before` builds; returns how many it built, and sets `skipped` when it was skipped
    /// from the loading screen.
    size_t PrecacheGraphicsWarm(u64 title_id, size_t num_threads, size_t done_before,
                                size_t total_before, std::stop_token stop_loading,
                                const VideoCore::DiskResourceLoadCallback& callback,
                                bool& skipped);

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};
    const size_t num_pipeline_workers;
    EmissionBudget emission_budget;

    GraphicsPipelineCacheKey graphics_key{};
    GraphicsPipeline* current_pipeline{};

    /// EDEN_NX2_GPL: shader libraries shared by the graphics pipelines. Declared before the
    /// pipelines, so it outlives them.
    GraphicsPipelineLibraryCache graphics_libraries;

    /// Destroys graphics pipelines unused for a while, once the GPU is done with them.
    void EvictUnusedPipelines();
    u64 frame_count{};
    /// Wait budget of native draws for unbuilt pipelines left in this frame, in microseconds
    /// (native_async_draws = Bounded). Only touched on the GPU thread; reset in TickFrame.
    s64 native_wait_budget_us{8000};
    s64 native_wait_used_us{}; ///< Budget spent since the last 10 s log line.

    struct TargetFrames {
        u64 last;     ///< Last frame the target was drawn to
        u64 previous; ///< The frame before that one it was drawn to
    };
    std::unordered_map<u64, TargetFrames> target_frames; ///< Render target key -> frames
    u64 last_target_key{~0ULL};
    u64 last_target_frame{~0ULL};
    bool last_target_redrawn{};
    enum class TargetReason { Redrawn, NotCleared, NewTarget };
    TargetReason last_target_reason{TargetReason::NotCleared};
    std::unordered_map<u64, u64> cleared_frames; ///< Target address -> last frame it was cleared
    std::deque<std::pair<u64, std::unique_ptr<GraphicsPipeline>>> evicted_pipelines; ///< (tick, pipeline)
    std::unordered_set<u64> used_and_evicted; ///< Key hashes of pipelines used this session, evicted since

    ::Common::unordered_map<ComputePipelineCacheKey, ComputePipelineCacheEntry> compute_cache;
    /// Replaced compute pipelines, destroyed once the GPU and their queued build are done with
    /// them (tick, pipeline)
    std::deque<std::pair<u64, std::unique_ptr<ComputePipeline>>> retired_compute_pipelines;
    ::Common::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;

    ShaderPools main_pools;
    std::mutex spare_pools_mutex;
    std::vector<std::shared_ptr<ShaderPools>> spare_pools;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;

    /// Key hashes of the graphics pipelines in the disk cache. Only those a recent session used
    /// (used_pipelines) are built at boot; the rest are built when first needed, and not stored
    /// again. Each built pipeline holds driver memory (title 1: 6.5k cached, a quarter used).
    std::unordered_set<u64> disk_pipelines;
    std::unordered_map<u64, u8> used_pipelines; ///< Key hash -> sessions since it was last used
    std::unordered_set<const GraphicsPipeline*> used_this_session;
    std::filesystem::path used_pipelines_filename;
    std::chrono::steady_clock::time_point used_pipelines_saved{};
    void SaveUsedPipelines();

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;
    size_t pipelines_since_flush{};
    std::chrono::steady_clock::time_point last_flush{};
    std::atomic<size_t> last_cache_size{};
    std::atomic_bool flush_in_flight{};

    /// Requested first when the cache is destroyed, to stop a rewrite of the disk cache: Android
    /// and yuzu_cmd load with a stop token that never stops
    std::stop_source serialization_stop;
    Common::ThreadWorker workers;
    Common::ThreadWorker serialization_thread;
    DynamicFeatures dynamic_features;
    PipelinePrebuild<GraphicsPipelineCacheKey, GraphicsPipeline> background_pipelines;
};

} // namespace Vulkan
