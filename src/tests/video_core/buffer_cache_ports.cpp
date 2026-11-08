// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "common/page_table.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/buffer_cache/upload_order.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu.h"
#include "video_core/nx2_list_stats.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace {

struct Runtime;

struct Buffer : VideoCommon::BufferBase {
    Buffer(Runtime&, VideoCommon::NullBufferParams params) : BufferBase{params} {}
    Buffer(Runtime&, DAddr address, u32 size, bool mapped)
        : BufferBase{address, size}, bytes(size), host_mapped{mapped} {}

    void MarkUsage(u64 offset, u64 size) {
        REQUIRE(offset + size <= bytes.size());
    }
    std::span<u8> Mapped() {
        return host_mapped ? std::span{bytes} : std::span<u8>{};
    }
    u64 LastUsageTick() const {
        return tick;
    }

    std::vector<u8> bytes;
    bool host_mapped{};
    u64 tick{};
};

struct Staging {
    Buffer* buffer;
    u64 offset;
    std::span<u8> mapped_span;
};

struct Runtime {
    bool CanReportMemoryUsage() const {
        return false;
    }
    u64 GetDeviceLocalMemory() const {
        return 0;
    }
    u64 GetMaxStorageBufferRange() const {
        return 1ULL << 30;
    }
    u32 GetStorageBufferAlignment() const {
        return alignment;
    }
    u32 GetUniformBufferAlignment() const {
        return 1;
    }
    bool IsFree(u64 tick) const {
        return tick <= completed_tick;
    }
    u64 CurrentTick() const {
        return submitted_tick + 1;
    }
    void WaitSubmittedTick(u64) {}
    VideoCommon::UploadOrder DecideUploadOrder(const Buffer&,
                                               std::span<VideoCommon::BufferCopy>) const {
        return {};
    }
    Staging DownloadStagingBuffer(u64 size, bool = false) {
        // Kept until the end: read downloads hold theirs past the next request.
        stagings.push_back(std::make_unique<Buffer>(*this, 0, size + 128, false));
        ++allocations;
        Buffer* const staging = stagings.back().get();
        return {staging, 128, std::span{staging->bytes}.subspan(128)};
    }
    Staging UploadStagingBuffer(u64 size) {
        return DownloadStagingBuffer(size);
    }
    void FreeDeferredStagingBuffer(Staging&) {}
    /// Buffer to buffer, as when a new buffer takes over the ones it overlaps.
    void CopyBuffer(Buffer& dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src.bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst.bytes.size());
            std::memcpy(dst.bytes.data() + copy.dst_offset, src.bytes.data() + copy.src_offset,
                        copy.size);
        }
    }
    void ClearBuffer(Buffer& buffer, u32 offset, size_t size, u32 value) {
        REQUIRE(offset + size <= buffer.bytes.size());
        for (size_t i = 0; i + sizeof(value) <= size; i += sizeof(value)) {
            std::memcpy(buffer.bytes.data() + offset + i, &value, sizeof(value));
        }
    }
    void CopyBuffer(Buffer* dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src.bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst->bytes.size());
            pending.push_back([dst, &src, copy] {
                std::memcpy(dst->bytes.data() + copy.dst_offset, src.bytes.data() + copy.src_offset,
                            copy.size);
            });
            ++downloads;
        }
    }
    void CopyBuffer(Buffer& dst, Buffer* src, std::span<const VideoCommon::BufferCopy> copies, bool,
                    VideoCommon::UploadOrder) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src->bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst.bytes.size());
            std::memcpy(dst.bytes.data() + copy.dst_offset, src->bytes.data() + copy.src_offset,
                        copy.size);
            ++uploads;
            upload_bytes += copy.size;
        }
    }
    void Finish() {
        if (before_completion) {
            before_completion();
        }
        Complete(submitted_tick);
        for (auto& copy : pending) {
            copy();
        }
        pending.clear();
        ++finishes;
    }
    u64 SubmitForHostRead() {
        ++host_read_barriers;
        submitted.emplace_back(++submitted_tick, std::move(pending));
        pending.clear();
        return submitted_tick;
    }
    u64 QueueForHostRead() {
        ++queued_host_reads;
        submitted.emplace_back(++submitted_tick, std::move(pending));
        pending.clear();
        return submitted_tick;
    }
    bool IsFreeNow(u64 tick) const {
        return IsFree(tick);
    }
    void WaitForHostRead(u64 tick) {
        if (before_completion) {
            before_completion();
        }
        Complete(tick);
        ++host_read_waits;
    }
    /// The host GPU runs what was submitted up to `tick`.
    void Complete(u64 tick) {
        while (!submitted.empty() && submitted.front().first <= tick) {
            for (auto& copy : submitted.front().second) {
                copy();
            }
            submitted.pop_front();
        }
        completed_tick = (std::max)(completed_tick, tick);
    }
    void Wait(u64 tick) {
        if (before_completion) {
            before_completion();
        }
        waited_tick = tick;
        ++waits;
    }
    void FinishForHostRead() {
        ++host_read_barriers;
        Finish();
    }
    void PreCopyBarrier() {}
    void PostCopyBarrier() {}
    void BindUniformBuffer(Buffer&, u32, u32) {}
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        stream.resize(size);
        return stream;
    }

    u32 alignment{256};
    u32 allocations{}, downloads{}, uploads{}, finishes{}, waits{};
    u32 host_read_barriers{}, host_read_waits{}, queued_host_reads{};
    u64 upload_bytes{}, waited_tick{};
    u64 submitted_tick{}, completed_tick{};
    std::vector<std::unique_ptr<Buffer>> stagings;
    std::vector<u8> stream;
    std::vector<std::function<void()>> pending;
    std::deque<std::pair<u64, std::vector<std::function<void()>>>> submitted;
    std::function<void()> before_completion;
};

struct Tracker : VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager> {
    using MemoryTrackerBase::MemoryTrackerBase;
    void MarkRegionAsCpuModified(DAddr address, u64 size) {
        ++cpu_marks;
        MemoryTrackerBase::MarkRegionAsCpuModified(address, size);
    }
    u32 cpu_marks{};
};

struct Policy {
    using Runtime = ::Runtime;
    using Buffer = ::Buffer;
    using Async_Buffer = Staging;
    using MemoryTracker = Tracker;
    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using Cache = VideoCommon::BufferCache<Policy>;

} // namespace

namespace VideoCommon {

struct BufferCacheTestAccess {
    static Binding Bind(Cache& cache, Tegra::MemoryManager& memory, GPUVAddr descriptor,
                        bool written = false, bool native = false, bool table = false) {
        cache.gpu_memory = &memory;
        return cache.StorageBufferBinding(descriptor, 0, written, native, table);
    }
    static BufferId Add(Cache& cache, DAddr address, u32 size, bool mapped = false) {
        const auto id = cache.slot_buffers.insert(cache.runtime, address, size, mapped);
        cache.Register(id);
        cache.memory_tracker.ForEachUploadRange(address, size, [](auto, auto) {});
        return id;
    }
    static Buffer& Get(Cache& cache, BufferId id) {
        return cache.slot_buffers[id];
    }
    static void Dirty(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsGpuModified(address, size);
        cache.gpu_modified_ranges.Add(address, size);
        cache.uncommitted_gpu_modified_ranges.Add(address, size);
        cache.committed_gpu_modified_ranges.emplace_back().Add(address, size);
        cache.pending_download_ranges.emplace_back().Add(address, size);
    }
    static bool Pending(Cache& cache, DAddr address, u64 size) {
        if (cache.gpu_modified_ranges.Intersects(address, size) ||
            cache.uncommitted_gpu_modified_ranges.Intersects(address, size)) {
            return true;
        }
        for (const auto& ranges : cache.committed_gpu_modified_ranges) {
            if (ranges.Intersects(address, size))
                return true;
        }
        for (const auto& ranges : cache.pending_download_ranges) {
            if (ranges.Intersects(address, size))
                return true;
        }
        return false;
    }
    static void DrawWrite(Cache& cache, DAddr page) {
        std::scoped_lock lock{cache.nx2_list_mutex};
        auto& entry = cache.nx2_list_pages[page];
        entry.draw_written = true;
        ++entry.writes;
    }
    static void Download(Cache& cache, std::span<const std::pair<DAddr, u64>> ranges) {
        cache.DownloadRangesMemory(ranges);
    }
    /// A draw or dispatch binds [address, address + size) of buffer `id`: its CPU-modified bytes
    /// are uploaded.
    static void Upload(Cache& cache, BufferId id, DAddr address, u32 size) {
        (void)cache.SynchronizeBuffer(cache.slot_buffers[id], address, size);
    }
    /// A guest write the buffer cache doesn't hear of, on a page the tracker counts as written.
    static void MarkCpuModified(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsCpuModified(address, size);
    }
    static void Uniform(Cache& cache, BufferCacheChannelInfo& channel, BufferId id, DAddr address,
                        u32 size) {
        cache.channel_state = &channel;
        cache.maxwell3d = &channel.maxwell3d;
        cache.gpu_memory = &channel.gpu_memory;
        channel.uniform_buffers[0][0] = {.device_addr = address, .size = size, .buffer_id = id};
        cache.BindHostGraphicsUniformBuffer(0, 0, 0, true);
    }
    static void NextFrame(Cache& cache) {
        cache.native_indirect_uniforms_this_frame.clear();
    }
    static u32 CpuMarks(Cache& cache) {
        return cache.memory_tracker.cpu_marks;
    }
    static void MarkWritten(Cache& cache, BufferId id, DAddr address, u32 size) {
        cache.MarkWrittenBuffer(id, address, size);
    }
    /// A dispatch or copy wrote [address, address + size) of buffer `id` (Nx2NoteListWrite).
    static void ListWrite(Cache& cache, BufferId id, DAddr address, u64 size) {
        cache.Nx2NoteListWrite(id, address, size);
    }
    static void UseGpuMemory(Cache& cache, Tegra::MemoryManager& memory) {
        cache.gpu_memory = &memory;
    }
    static size_t ReadDownloads(Cache& cache) {
        return cache.read_downloads.size();
    }
    /// TEST (EDEN_NX2_READ_CHECK): every read download compared with a synchronous flush.
    static void CheckEveryRead(Cache& cache) {
        cache.read_check.every = 1;
    }
    static const auto& ReadChecks(Cache& cache) {
        return cache.read_check;
    }
};

} // namespace VideoCommon

namespace {

using Access = VideoCommon::BufferCacheTestAccess;

struct Fixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    std::optional<Tegra::MemoryManager> memory;
    Runtime runtime;
    std::unique_ptr<Cache> cache;
    std::unique_ptr<Kernel::KProcess> process;
    Core::Asid asid{};
    u8* bytes{};

    Fixture() {
        system.Initialize();
        process = std::make_unique<Kernel::KProcess>(system.Kernel());
        auto& table = process->GetPageTable().GetImpl();
        table.Resize(24, 12);
        bytes = physical.GetPointerFromRaw<u8>(0x10000);
        std::memset(bytes, 0, 0x40000);
        for (u64 page = 0; page < 64; ++page) {
            table.entries.GetAndFault(0x10 + page)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(bytes) - 0x10000);
        }
        process->GetMemory().SetCurrentPageTable(*process);
        asid = device.RegisterProcess(&process->GetMemory());
        device.BindInterface(&rasterizer);
        device.Map(0x10000, 0x10000, 0x40000, asid);
        ResetMemory();
        cache = std::make_unique<Cache>(device, runtime);
    }
    ~Fixture() {
        cache.reset();
        memory.reset();
        device.Unmap(0x10000, 0x40000);
        device.UnregisterProcess(asid);
        system.Kernel().UnregisterKernelObject(process.get());
    }
    void ResetMemory() {
        memory.reset();
        memory.emplace(system, device, 32, 1ULL << 31, 16);
        memory->BindRasterizer(&rasterizer);
        memory->Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
    }
    void Descriptor(GPUVAddr address, u64 size) {
        std::memcpy(bytes, &address, sizeof(address));
        std::memcpy(bytes + 8, &size, sizeof(size));
    }
    VideoCommon::Binding Bind(bool written = false, bool native = false, bool table = false) {
        return Access::Bind(*cache, *memory, 0x100000, written, native, table);
    }
};

/// Hands FinishRead to a buffer cache, as RasterizerVulkan does.
class ReadRasterizer final : public VideoCore::RasterizerInterface {
public:
    explicit ReadRasterizer(Cache& cache_) : cache{cache_} {}

    void FinishRead(DAddr addr, u64 size) override {
        cache.FinishReadDownloads(addr, size);
    }
    void Draw(bool, u32) override {}
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&&) override {}
    void SyncOperation(std::function<void()>&&) override {}
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override {
        return false;
    }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, true};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(DAddr, u64) override {}
    bool OnCPUWrite(DAddr, u64) override {
        return false;
    }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {}
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override {
        std::terminate();
    }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}

private:
    Cache& cache;
};

} // namespace

TEST_CASE("Storage binding cache tracks map changes and descriptor dependencies",
          "[buffer_ports]") {
    Fixture f;
    constexpr GPUVAddr base = 0x200000;
    f.memory->Map(base, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.Descriptor(base + 0x123, 0x456);
    for (int i = 0; i < 3; ++i) {
        const auto binding = f.Bind(true);
        REQUIRE(binding.device_addr == 0x20100);
        REQUIRE(binding.gpu_addr == base + 0x100);
        REQUIRE(binding.size == 0x479);
        REQUIRE(binding.buffer_id == VideoCommon::BufferId{});
        REQUIRE(binding.segment_count == 0);
    }
    REQUIRE(f.Bind().size == 0xf00);
    f.Descriptor(base + 0x123, 0x1456);
    REQUIRE(f.Bind(true).size == 0x1479);
    f.runtime.alignment = 4096;
    REQUIRE(f.Bind(true).device_addr == 0x20000);
    REQUIRE(f.Bind(true).size == 0x1579);
    REQUIRE(f.Bind(true, true).size == 0x10000);
    f.memory->Map(base, 0x30000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(f.Bind(true).device_addr == 0x30000);
    f.memory->MapSparse(base, 0x10000, true);
    REQUIRE(f.Bind().size == 0);
    f.memory->Unmap(base, 0x10000);
    REQUIRE(f.Bind().size == 0);
    f.memory->Map(base, 0x20000, 0x2000, Tegra::PTEKind::PITCH, false);
    REQUIRE(f.Bind(true, true).size == 0x2000);
    f.memory->Map(base + 0x1000, 0x40000, 0x1000, Tegra::PTEKind::PITCH, false);
    REQUIRE(f.Bind(true, true).size == 0x1000);
    const auto old_owner = f.memory->GetID();
    f.ResetMemory();
    f.memory->Map(base, 0x40000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(f.memory->GetID() != old_owner);
    REQUIRE(f.Bind(true).device_addr == 0x40000);
    f.Descriptor(0, 0x123);
    REQUIRE(f.Bind().size == 0);
}

TEST_CASE("Storage binding cache preserves native and table size policies", "[buffer_ports]") {
    Fixture f;
    constexpr GPUVAddr base = 0x4000000;
    f.memory->Map(base, 0x10000000, 80 << 20, Tegra::PTEKind::PITCH, true);
    f.Descriptor(base, 256);
    REQUIRE(f.Bind().size == 4096);
    REQUIRE(f.Bind(false, true).size == 8 << 20);
    REQUIRE(f.Bind(false, false, true).size == 80 << 20);
    REQUIRE(f.Bind(true, false, true).size == 256);
    f.Descriptor(base, (1ULL << 32) | 256);
    REQUIRE(f.Bind().size == 8 << 20);
    REQUIRE(f.Bind(false, false, true).size == 80 << 20);
    f.Descriptor(base + (79 << 20), 0);
    REQUIRE(f.Bind(false, false, true).size == 1 << 20);
}

TEST_CASE("Storage binding cache does not reuse a destroyed address space", "[buffer_ports]") {
    Fixture f;
    f.memory->Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.Descriptor(0x200000, 0x1000);
    REQUIRE(f.Bind(true).device_addr == 0x20000);
    const auto serial = f.memory->MapSerial();
    const auto owner = f.memory->GetID();
    const auto* address = &*f.memory;
    f.ResetMemory();
    f.memory->Map(0x200000, 0x30000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(&*f.memory == address);
    REQUIRE(f.memory->MapSerial() == serial);
    REQUIRE(f.memory->GetID() != owner);
    REQUIRE(f.Bind(true).device_addr == 0x30000);
}

TEST_CASE("Batch downloads clamp buffers, deduplicate pages and finish before publication",
          "[buffer_ports]") {
    Fixture f;
    const auto first = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto second = Access::Add(*f.cache, 0x30000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, first).bytes, 0x12);
    std::ranges::fill(Access::Get(*f.cache, second).bytes, 0x34);
    Access::Dirty(*f.cache, 0x2f000, 0x2000);
    Access::Dirty(*f.cache, 0x32000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.bytes[0x1f000] == 0);
        REQUIRE(f.bytes[0x20000] == 0);
    };
    const std::array<std::pair<DAddr, u64>, 3> ranges{{
        {0x2f008, 0x1000},
        {0x2f000, 0x2000},
        {0x20000, 0},
    }};
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.allocations == 1);
    REQUIRE(f.runtime.downloads == 2);
    REQUIRE(f.bytes[0x1f000] == 0x12);
    REQUIRE(f.bytes[0x1ffff] == 0x12);
    REQUIRE(f.bytes[0x20000] == 0x34);
    REQUIRE(f.bytes[0x20fff] == 0x34);
    REQUIRE(f.bytes[0x22000] == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x2f000, 0x2000));
    REQUIRE(Access::Pending(*f.cache, 0x32000, 0x1000));
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.finishes == 1);
}

TEST_CASE("Mapped and mixed batches make device writes host visible in one completion",
          "[buffer_ports]") {
    Fixture f;
    const auto first = Access::Add(*f.cache, 0x20000, 0x10000, true);
    const auto second = Access::Add(*f.cache, 0x30000, 0x10000, true);
    const auto clean = Access::Add(*f.cache, 0x40000, 0x10000, true);
    auto& a = Access::Get(*f.cache, first);
    auto& b = Access::Get(*f.cache, second);
    a.tick = 8;
    b.tick = 19;
    Access::Get(*f.cache, clean).tick = 99;
    const std::array<std::pair<DAddr, u64>, 3> ranges{{
        {0x20000, 0x1000},
        {0x30000, 0x1000},
        {0x40000, 0x1000},
    }};
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::Dirty(*f.cache, 0x30000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        REQUIRE(f.bytes[0x20000] == 0);
        std::ranges::fill(a.bytes, 0x56);
        std::ranges::fill(b.bytes, 0x78);
    };
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.host_read_barriers == 1);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.allocations == 0);
    REQUIRE(f.bytes[0x10000] == 0x56);
    REQUIRE(f.bytes[0x20000] == 0x78);
    b.host_mapped = false;
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::Dirty(*f.cache, 0x30000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 2);
        REQUIRE(f.bytes[0x10000] == 0x56);
        REQUIRE(f.bytes[0x20000] == 0x78);
        std::ranges::fill(a.bytes, 0x9a);
        std::ranges::fill(b.bytes, 0xbc);
    };
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.finishes == 2);
    REQUIRE(f.runtime.downloads == 1);
    REQUIRE(f.bytes[0x10000] == 0x9a);
    REQUIRE(f.bytes[0x20000] == 0xbc);
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.host_read_barriers == 2);
    REQUIRE(f.runtime.finishes == 2);
}

TEST_CASE("Single mapped download completes host visibility before guest publication",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, true);
    auto& buffer = Access::Get(*f.cache, id);
    buffer.tick = 7;
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        std::ranges::fill(buffer.bytes, 0xcd);
    };
    f.cache->DownloadMemory(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.allocations == 0);
    REQUIRE(f.bytes[0x10000] == 0xcd);
    f.cache->DownloadMemory(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 1);
}

TEST_CASE("Command page batch publishes only completed matching write generations",
          "[buffer_ports]") {
    Fixture f;
    const bool mapped = GENERATE(false, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x42);
    Access::Dirty(*f.cache, 0x20000, 0x3000);
    Access::DrawWrite(*f.cache, 0x20000);
    Access::DrawWrite(*f.cache, 0x21000);
    Access::DrawWrite(*f.cache, 0x22000);
    std::array<u8, 4> out{};
    using Read = Cache::Nx2ListRead;
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        REQUIRE(f.cache->Nx2ReadListCopy(0x20000, out, true) == Read::Unknown);
        Access::DrawWrite(*f.cache, 0x22000);
    };
    const std::array<DAddr, 1> requested{0x20000};
    f.cache->Nx2FlushListPages(requested);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.downloads == (mapped ? 0U : 3U));
    REQUIRE(f.cache->Nx2ReadListCopy(0x20000, out, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x22000, out, true) == Read::Unknown);
    REQUIRE(f.bytes[0x12000] == 0x42);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, false) == Read::NeedsSync);
    Access::DrawWrite(*f.cache, 0x21000);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, true) == Read::Unknown);
}

TEST_CASE("Command page readers remain ordered during batch completion", "[buffer_ports]") {
    struct {
        Core::DeviceMemory physical;
        Tegra::MaxwellDeviceMemoryManager device{physical};
        Runtime runtime;
        std::unique_ptr<Cache> cache = std::make_unique<Cache>(device, runtime);
    } f;
    Access::Add(*f.cache, 0x20000, 0x10000);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::DrawWrite(*f.cache, 0x20000);
    std::barrier rendezvous{2};
    Cache::Nx2ListRead before{}, after{};
    std::jthread reader{[&] {
        std::array<u8, 4> out{};
        rendezvous.arrive_and_wait();
        before = f.cache->Nx2ReadListCopy(0x20000, out, true);
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
        after = f.cache->Nx2ReadListCopy(0x20000, out, true);
    }};
    f.runtime.before_completion = [&] {
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
    };
    const std::array<DAddr, 1> requested{0x20000};
    f.cache->Nx2FlushListPages(requested);
    rendezvous.arrive_and_wait();
    reader.join();
    REQUIRE(before == Cache::Nx2ListRead::Unknown);
    REQUIRE(after == Cache::Nx2ListRead::Synced);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.downloads == 1);
}

namespace {

/// Makes `page` a command list page for one test (nx2.h keeps them for the session).
struct ListPage {
    explicit ListPage(DAddr page) {
        Tegra::Nx2ClearGpuWrittenPages();
        Tegra::Nx2MarkCommandListPage(page);
    }
    ~ListPage() {
        Tegra::Nx2ClearGpuWrittenPages();
    }
    ListPage(const ListPage&) = delete;
    ListPage& operator=(const ListPage&) = delete;
};

bool AllBytes(std::span<const u8> bytes, u8 value) {
    return std::ranges::all_of(bytes, [value](u8 byte) { return byte == value; });
}

} // namespace

TEST_CASE("Command lists on words no GPU work wrote read guest memory, not the page's copy",
          "[buffer_ports]") {
    Fixture f;
    const ListPage list_page{0x20000};
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x77);
    std::memset(f.bytes + 0x10000, 0x11, 0x1000);
    // The page's first list drained the split and flushed the page (FlushGpuWrittenCommands).
    f.cache->Nx2NoteListSynced(0x20000);
    // A dispatch writes bytes 0x800-0x900 of the page; the copy taken after it is on its way.
    f.cache->Nx2CollectListCopies(true);
    Access::ListWrite(*f.cache, id, 0x20800, 0x100);
    REQUIRE(f.cache->Nx2CopyListPages(f.runtime.CurrentTick()));
    f.cache->Nx2CollectListCopies(false);
    using Read = Cache::Nx2ListRead;
    std::array<u8, 16> list{};
    REQUIRE(f.cache->Nx2ReadListCopy(0x20100, list, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x207f0, list, true) == Read::Synced);
    // The GPU completes the copy. A list on the dispatch's words reads them from it; one that
    // starts before them reads its other words from guest memory.
    f.runtime.Finish();
    REQUIRE(f.cache->Nx2ReadListCopy(0x20800, list, true) == Read::Copied);
    REQUIRE(AllBytes(list, 0x77));
    std::array<u8, 32> spanning{};
    REQUIRE(f.cache->Nx2ReadListCopy(0x207f0, spanning, true) == Read::Copied);
    REQUIRE(AllBytes(std::span{spanning}.first(16), 0x11));
    REQUIRE(AllBytes(std::span{spanning}.last(16), 0x77));
}

TEST_CASE("Command list pages the front has not flushed read every word from the copy",
          "[buffer_ports]") {
    Fixture f;
    const ListPage list_page{0x20000};
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x77);
    std::memset(f.bytes + 0x10000, 0x11, 0x1000);
    // GPU work may have written the page before it became a command list page: only the first
    // write since is noted, and guest memory need not hold the rest.
    f.cache->Nx2CollectListCopies(true);
    Access::ListWrite(*f.cache, id, 0x20800, 0x100);
    REQUIRE(f.cache->Nx2CopyListPages(f.runtime.CurrentTick()));
    f.cache->Nx2CollectListCopies(false);
    f.runtime.Finish();
    std::array<u8, 32> list{};
    REQUIRE(f.cache->Nx2ReadListCopy(0x20100, list, true) == Cache::Nx2ListRead::Copied);
    REQUIRE(AllBytes(list, 0x77));
    REQUIRE(f.cache->Nx2ReadListCopy(0x207f0, list, true) == Cache::Nx2ListRead::Copied);
    REQUIRE(AllBytes(list, 0x77));
}

TEST_CASE("Host writes give command list words back to guest memory and retire the page's copy",
          "[buffer_ports]") {
    Fixture f;
    const ListPage list_page{0x20000};
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    f.cache->Nx2NoteListSynced(0x20000);
    f.cache->Nx2CollectListCopies(true);
    Access::ListWrite(*f.cache, id, 0x20800, 0x100);
    REQUIRE(f.cache->Nx2CopyListPages(f.runtime.CurrentTick()));
    f.cache->Nx2CollectListCopies(false);
    f.runtime.Finish();
    using Read = Cache::Nx2ListRead;
    std::array<u8, 16> list{};
    REQUIRE(f.cache->Nx2ReadListCopy(0x20840, list, true) == Read::Copied);
    // The CPU, an inline upload or a copy of bytes guest memory holds writes 0x800-0x840 whole,
    // and two bytes of each of the words at 0x880 and 0x884.
    f.cache->Nx2NoteListHostWrite(0x20800, 0x40);
    f.cache->Nx2NoteListHostWrite(0x20882, 4);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20800, list, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20830, list, true) == Read::Synced);
    // GPU-written words it did not write whole: the copy no longer holds the page.
    REQUIRE(f.cache->Nx2ReadListCopy(0x20840, list, true) == Read::Unknown);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20880, list, true) == Read::Unknown);
}

TEST_CASE("A flushed command list page reads guest memory until GPU work writes it again",
          "[buffer_ports]") {
    Fixture f;
    const ListPage list_page{0x20000};
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    f.cache->Nx2CollectListCopies(true);
    Access::ListWrite(*f.cache, id, 0x20000, 0x1000);
    REQUIRE(f.cache->Nx2CopyListPages(f.runtime.CurrentTick()));
    using Read = Cache::Nx2ListRead;
    std::array<u8, 16> list{};
    REQUIRE(f.cache->Nx2ReadListCopy(0x20800, list, true) == Read::Copied);
    // The front drained the split and flushed the page (DmaPusher::FlushGpuWrittenCommands).
    f.cache->Nx2NoteListSynced(0x20000);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20800, list, true) == Read::Synced);
    Access::ListWrite(*f.cache, id, 0x20900, 0x40);
    REQUIRE(f.cache->Nx2CopyListPages(f.runtime.CurrentTick()));
    f.cache->Nx2CollectListCopies(false);
    f.runtime.Finish();
    REQUIRE(f.cache->Nx2ReadListCopy(0x20800, list, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20900, list, true) == Read::Copied);
}

TEST_CASE("Copy engine clears give command list words back to guest memory", "[buffer_ports]") {
    Fixture f;
    const ListPage list_page{0x20000};
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&f.rasterizer);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    Access::UseGpuMemory(*f.cache, gpu_memory);
    f.cache->Nx2NoteListSynced(0x20000);
    f.cache->Nx2CollectListCopies(true);
    Access::ListWrite(*f.cache, id, 0x20800, 0x100);
    REQUIRE(f.cache->Nx2CopyListPages(f.runtime.CurrentTick()));
    f.cache->Nx2CollectListCopies(false);
    f.runtime.Finish();
    // 16 words cleared (MaxwellDMA's remapped constant fill writes them to guest memory too).
    REQUIRE(f.cache->DMAClear(0x200800, 16, 0));
    REQUIRE(AllBytes(std::span{Access::Get(*f.cache, id).bytes}.subspan(0x800, 0x40), 0));
    using Read = Cache::Nx2ListRead;
    std::array<u8, 16> list{};
    REQUIRE(f.cache->Nx2ReadListCopy(0x20830, list, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x20840, list, true) == Read::Unknown);
}

TEST_CASE("EDEN_NX2_LIST_VERIFY reads command lists as before, from the page's whole copy",
          "[buffer_ports]") {
    Fixture f;
    const ListPage list_page{0x20000};
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x77);
    std::memset(f.bytes + 0x10000, 0x11, 0x1000);
    f.cache->Nx2NoteListSynced(0x20000);
    f.cache->Nx2CollectListCopies(true);
    Access::ListWrite(*f.cache, id, 0x20800, 0x100);
    REQUIRE(f.cache->Nx2CopyListPages(f.runtime.CurrentTick()));
    f.cache->Nx2CollectListCopies(false);
    f.runtime.Finish();
    using Read = Cache::Nx2ListRead;
    std::array<u8, 32> list{};
    {
        // Whatever words GPU work wrote: once it wrote the page, every list there reads the copy.
        const Tegra::Nx2ListStats::ReferenceScope reference;
        REQUIRE(f.cache->Nx2ReadListCopy(0x20100, list, true) == Read::Copied);
        REQUIRE(AllBytes(list, 0x77));
        REQUIRE(f.cache->Nx2ReadListCopy(0x207f0, list, true) == Read::Copied);
        REQUIRE(AllBytes(list, 0x77));
    }
    REQUIRE(f.cache->Nx2ReadListCopy(0x20100, list, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x207f0, list, true) == Read::Copied);
    REQUIRE(AllBytes(std::span{list}.first(16), 0x11));
}

TEST_CASE("Native uniform banks upload directly without overwriting GPU-owned bytes",
          "[buffer_ports]") {
    Fixture f;
    f.memory->EnableNativeTiling();
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x10000;
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::memset(f.bytes + 0x10000, 0x5a, 0x10000);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x10000);
    REQUIRE(Access::CpuMarks(*f.cache) == 0);
    REQUIRE(Access::Get(*f.cache, id).bytes[0xffff] == 0x5a);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.uploads == 1);
    Access::NextFrame(*f.cache);
    Access::Dirty(*f.cache, 0x21000, 0x1000);
    std::memset(f.bytes + 0x10000, 0xa5, 0x10000);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.uploads == 1);
    REQUIRE(Access::Get(*f.cache, id).bytes[0x1000] == 0x5a);
}

TEST_CASE("Native uniform uploads stop at holes and keep unaligned bank boundaries",
          "[buffer_ports]") {
    Fixture f;
    f.memory->EnableNativeTiling();
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x10000;
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x20000);
    std::memset(f.bytes + 0x10000, 0x6b, 0x20000);
    f.device.Unmap(0x22000, 0x1000);
    Access::Uniform(*f.cache, channel, id, 0x20123, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x1edd);
    const auto& data = Access::Get(*f.cache, id).bytes;
    REQUIRE(data[0x122] == 0);
    REQUIRE(data[0x123] == 0x6b);
    REQUIRE(data[0x1fff] == 0x6b);
    REQUIRE(data[0x2000] == 0);
    REQUIRE(data[0x3000] == 0);
    REQUIRE(Access::CpuMarks(*f.cache) == 0);
    f.device.Map(0x22000, 0x22000, 0x1000, f.asid);
    Access::NextFrame(*f.cache);
    Access::Uniform(*f.cache, channel, id, 0x20123, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x11edd);
    REQUIRE(data[0x10122] == 0x6b);
    REQUIRE(data[0x10123] == 0);
}

TEST_CASE("Downloads keep CPU writes collected and not yet applied", "[buffer_ports]") {
    Fixture f;
    const auto staged = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto mapped = Access::Add(*f.cache, 0x30000, 0x10000, true);
    std::ranges::fill(Access::Get(*f.cache, staged).bytes, 0x12);
    std::ranges::fill(Access::Get(*f.cache, mapped).bytes, 0x34);
    f.cache->SetPendingCpuWrites([](const std::function<void(DAddr, size_t)>& callback) {
        callback(0x20100, 0x10);
        callback(0x30ff8, 0x10);
    });
    std::memset(f.bytes + 0x10100, 0xee, 0x10);
    std::memset(f.bytes + 0x20ff8, 0xee, 0x10);
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    Access::Dirty(*f.cache, 0x30000, 0x2000);
    const std::array<std::pair<DAddr, u64>, 1> ranges{{{0x20000, 0x2000}}};
    Access::Download(*f.cache, ranges);
    REQUIRE(f.bytes[0x100ff] == 0x12);
    REQUIRE(std::all_of(f.bytes + 0x10100, f.bytes + 0x10110, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x10110] == 0x12);
    REQUIRE(f.bytes[0x11fff] == 0x12);
    f.cache->DownloadMemory(0x30000, 0x2000);
    REQUIRE(f.bytes[0x20ff7] == 0x34);
    REQUIRE(std::all_of(f.bytes + 0x20ff8, f.bytes + 0x21008, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x21008] == 0x34);
}

TEST_CASE("Read downloads stay GPU-modified until the reading thread writes them",
          "[buffer_ports]") {
    Fixture f;
    const bool mapped = GENERATE(false, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 0);
    REQUIRE(f.runtime.downloads == 1);
    REQUIRE(f.runtime.host_read_barriers == 1);
    REQUIRE(VideoCore::g_read_downloads == 1);
    REQUIRE(f.bytes[0x10000] == 0);
    REQUIRE(Access::Pending(*f.cache, 0x20000, 0x1000));
    // Another read of the range waits for the same copy, and its flush copies nothing.
    const auto area = f.cache->GetFlushArea(0x20800, 4);
    REQUIRE(area);
    REQUIRE_FALSE(area->preemtive);
    REQUIRE(area->in_flight);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(f.runtime.downloads == 1);
    REQUIRE(f.runtime.submitted_tick == 1);
    f.cache->FinishReadDownloads(0x20800, 4);
    REQUIRE(f.runtime.host_read_waits == 1);
    REQUIRE(VideoCore::g_read_downloads == 0);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
    REQUIRE(f.bytes[0x11000] == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
    REQUIRE(Access::Pending(*f.cache, 0x21000, 0x1000));
    REQUIRE(f.cache->GetFlushArea(0x20800, 4)->preemtive);
    f.cache->FinishReadDownloads(0x20000, 0x2000);
    REQUIRE(f.runtime.host_read_waits == 1);
}

TEST_CASE("Read downloads keep CPU writes made while they were on their way", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // After the submission: one write still collected, one already applied.
    std::memset(f.bytes + 0x10100, 0xee, 0x10);
    std::memset(f.bytes + 0x10200, 0xdd, 0x10);
    f.cache->SetPendingCpuWrites(
        [](const std::function<void(DAddr, size_t)>& callback) { callback(0x20100, 0x10); });
    const std::array<std::pair<DAddr, u64>, 1> applied{{{0x20200, 0x10}}};
    f.cache->ClaimCpuWrites(applied);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x100ff] == 0x12);
    REQUIRE(std::all_of(f.bytes + 0x10100, f.bytes + 0x10110, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x10110] == 0x12);
    REQUIRE(std::all_of(f.bytes + 0x10200, f.bytes + 0x10210, [](u8 b) { return b == 0xdd; }));
    REQUIRE(f.bytes[0x10210] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Read downloads land oldest first and leave newer GPU writes to download",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // The GPU thread's next work writes part of the range: on the host GPU, after the copy.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x100, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x100);
    const auto area = f.cache->GetFlushArea(0x20800, 4);
    REQUIRE_FALSE(area->preemtive);
    REQUIRE_FALSE(area->in_flight);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(f.runtime.downloads == 2);
    REQUIRE(Access::ReadDownloads(*f.cache) == 2);
    REQUIRE(f.cache->GetFlushArea(0x20800, 4)->in_flight);
    // The first copy holds what a synchronous flush would have left for the whole range.
    f.cache->FinishReadDownloads(0x20000, 4);
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x12);
    REQUIRE(Access::Pending(*f.cache, 0x20800, 0x100));
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x800));
    f.cache->FinishReadDownloads(0x20800, 4);
    REQUIRE(f.bytes[0x107ff] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x34);
    REQUIRE(f.bytes[0x108ff] == 0x34);
    REQUIRE(f.bytes[0x10900] == 0x12);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
    // Both written in order even when the newer one's reader comes first.
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x100, u8{0x56}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x100);
    f.cache->DownloadMemoryForRead(0x20800, 0x100);
    REQUIRE(Access::ReadDownloads(*f.cache) == 2);
    f.cache->FinishReadDownloads(0x20800, 4);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(f.bytes[0x10800] == 0x56);
    REQUIRE(f.bytes[0x10900] == 0x12);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Read downloads land before a later fence's downloads", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // A later GPU write the next fence downloads, and the host GPU runs all of it before anyone
    // waits for the read's copy.
    f.runtime.pending.push_back([&buffer] { std::fill_n(buffer.bytes.begin(), 0x100, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->CommitAsyncFlushes();
    f.runtime.Finish();
    f.cache->PopAsyncFlushes();
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x10000] == 0x34);
    REQUIRE(f.bytes[0x100ff] == 0x34);
    REQUIRE(f.bytes[0x10100] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Preflushed ranges wait for the read download still on its way", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    // The first read marks the range for fence downloads, then flushes it.
    const auto first = f.cache->GetFlushArea(0x20000, 4);
    REQUIRE_FALSE(first->preemtive);
    REQUIRE_FALSE(first->in_flight);
    f.cache->DownloadMemoryForRead(first->start_address, first->end_address - first->start_address);
    const auto second = f.cache->GetFlushArea(0x20010, 4);
    REQUIRE_FALSE(second->preemtive);
    REQUIRE(second->in_flight);
    f.cache->FinishReadDownloads(second->start_address,
                                 second->end_address - second->start_address);
    REQUIRE(f.bytes[0x10010] == 0x12);
    REQUIRE(f.cache->GetFlushArea(0x20010, 4)->preemtive);
}

TEST_CASE("GPU writes to guest memory land the read downloads under them first", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    ReadRasterizer reads{*f.cache};
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&reads);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // A semaphore release, then an inline upload, after the copy was taken.
    gpu_memory.Write<u32>(0x200100, 0xdeadbeef);
    REQUIRE(VideoCore::g_read_downloads == 0);
    const u32 words = 0x55555555;
    gpu_memory.WriteBlockUnsafe(0x200200, &words, sizeof(words));
    u32 value{};
    std::memcpy(&value, f.bytes + 0x10100, sizeof(value));
    REQUIRE(value == 0xdeadbeef);
    std::memcpy(&value, f.bytes + 0x10200, sizeof(value));
    REQUIRE(value == 0x55555555);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10104] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
}

TEST_CASE("Read downloads take again what a GPU write outside the cache's records changed",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // After the read's copy: a query result copy into the buffer (SyncValues), then a GPU write
    // elsewhere on the page, which the next read downloads.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{0x56}); });
    f.cache->OutdateReadDownloads(0x20040, 0x10);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x10, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x10);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x1003f] == 0x12);
    REQUIRE(f.bytes[0x10040] == 0x56);
    REQUIRE(f.bytes[0x1004f] == 0x56);
    REQUIRE(f.bytes[0x10050] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x34);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Switch 2 writes to guest memory land read downloads only on GPU-written pages",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    ReadRasterizer reads{*f.cache};
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&reads);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    struct Nx2Title {
        Nx2Title() {
            Tegra::g_nx2_ampere = true;
        }
        ~Nx2Title() {
            Tegra::g_nx2_ampere = false;
            Tegra::Nx2ClearGpuWrittenPages();
        }
    } nx2_title;
    Access::MarkWritten(*f.cache, id, 0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // A page no GPU write reached: no download can be on its way there.
    gpu_memory.Write<u32>(0x201000, 0xdeadbeef);
    REQUIRE(VideoCore::g_read_downloads == 1);
    REQUIRE(f.runtime.host_read_waits == 0);
    gpu_memory.Write<u32>(0x200100, 0xcafef00d);
    REQUIRE(VideoCore::g_read_downloads == 0);
    REQUIRE(f.runtime.host_read_waits == 1);
    u32 value{};
    std::memcpy(&value, f.bytes + 0x10100, sizeof(value));
    REQUIRE(value == 0xcafef00d);
    std::memcpy(&value, f.bytes + 0x11000, sizeof(value));
    REQUIRE(value == 0xdeadbeef);
    REQUIRE(f.bytes[0x10000] == 0x12);
    REQUIRE(f.bytes[0x10fff] == 0x12);
}

TEST_CASE("Read download check leaves out newer bytes over an earlier read's download",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::CheckEveryRead(*f.cache);
    ReadRasterizer reads{*f.cache};
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&reads);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    // The first read takes the page's GPU-written bytes, the second only those written since.
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x10, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x10);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(Access::ReadDownloads(*f.cache) == 2);
    // Constant buffer data over bytes the first read brings: that read lands first, and the
    // buffer cache hears of the write at the next FlushCaching. Synchronous flushes leave it too.
    std::array<u8, 0x30> words;
    words.fill(0xb6);
    gpu_memory.WriteBlockCached(0x200060, words.data(), words.size());
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x1005f] == 0x12);
    REQUIRE(f.bytes[0x10060] == 0xb6);
    REQUIRE(f.bytes[0x1008f] == 0xb6);
    REQUIRE(f.bytes[0x10090] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x34);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 2);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).rewritten == 0x30);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check finds an earlier read's bytes the GPU changed unrecorded",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    Access::CheckEveryRead(*f.cache);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    // A GPU write into the first read's bytes that nothing takes out of its download.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{0x56}); });
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x800, 0x10, u8{0x34}); });
    Access::MarkWritten(*f.cache, id, 0x20800, 0x10);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(f.bytes[0x10040] == 0x12);
    REQUIRE(f.bytes[0x10800] == 0x34);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 2);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0x10);
    REQUIRE(Access::ReadChecks(*f.cache).rewritten == 0);
}

namespace {
struct Nx2TitleScope {
    Nx2TitleScope() {
        Tegra::g_nx2_ampere = true;
    }
    ~Nx2TitleScope() {
        Tegra::g_nx2_ampere = false;
        Tegra::Nx2ClearGpuWrittenPages();
    }
};

/// A dispatch that writes `value` over [address, address + size) of the buffer at 0x20000.
void WriteByDispatch(Fixture& f, VideoCommon::BufferId id, DAddr address, u32 size, u8 value) {
    auto& buffer = Access::Get(*f.cache, id);
    f.runtime.pending.push_back([&buffer, address, size, value] {
        std::fill_n(buffer.bytes.begin() + static_cast<std::ptrdiff_t>(address - 0x20000), size,
                    value);
    });
    Access::MarkWritten(*f.cache, id, address, size);
    f.cache->NoteWorkRecorded(true);
}

/// A guest read of a word at `address`, as GPU::OnCPURead serves it.
VideoCore::RasterizerDownloadArea ReadWord(Fixture& f, DAddr address) {
    const auto area = f.cache->GetFlushArea(address, 4);
    REQUIRE(area);
    const u64 size = area->end_address - area->start_address;
    if (!area->preemtive && !area->in_flight) {
        f.cache->DownloadMemoryForRead(area->start_address, size);
    }
    if (!area->preemtive) {
        f.cache->FinishReadDownloads(area->start_address, size);
    }
    return *area;
}
} // namespace

TEST_CASE("Pages read after each GPU write get their downloads right behind the writes",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    // Learned from two writes the guest read; those reads still flush.
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        REQUIRE(f.runtime.queued_host_reads == 0);
        const auto area = ReadWord(f, 0x20010);
        REQUIRE_FALSE(area.preemtive);
        REQUIRE_FALSE(area.in_flight);
        REQUIRE(f.bytes[0x10010] == frame);
    }
    REQUIRE(f.runtime.host_read_waits == 2);
    // The next write's download goes right behind it, before any read.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    REQUIRE(f.runtime.queued_host_reads == 1);
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    REQUIRE(f.bytes[0x10010] == 2);
    // A read before the host GPU ran it waits for that copy alone.
    const auto waiting = f.cache->GetFlushArea(0x20010, 4);
    REQUIRE(waiting->in_flight);
    f.cache->FinishReadDownloads(waiting->start_address,
                                 waiting->end_address - waiting->start_address);
    REQUIRE(f.runtime.host_read_waits == 3);
    REQUIRE(f.runtime.host_read_barriers == 2);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE(f.bytes[0x100ff] == 3);
    // Once the host GPU ran it, the read finds the bytes in guest memory and waits for nothing.
    WriteByDispatch(f, id, 0x20000, 0x100, 4);
    REQUIRE(f.runtime.queued_host_reads == 2);
    f.runtime.Complete(f.runtime.submitted_tick);
    REQUIRE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.runtime.host_read_waits == 3);
    REQUIRE(f.bytes[0x10010] == 4);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(VideoCore::g_read_downloads == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Early downloads leave what a newer GPU write changed to a download of its own",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    // Another dispatch writes part of it again before the read: it gets an early download too.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    WriteByDispatch(f, id, 0x20080, 0x40, 4);
    REQUIRE(f.runtime.queued_host_reads == 2);
    f.runtime.Complete(f.runtime.submitted_tick);
    REQUIRE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x1007f] == 3);
    REQUIRE(f.bytes[0x10080] == 4);
    REQUIRE(f.bytes[0x100bf] == 4);
    REQUIRE(f.bytes[0x100c0] == 3);
    // A draw writes part of it again: the read downloads that part after the early download.
    WriteByDispatch(f, id, 0x20000, 0x100, 5);
    auto& buffer = Access::Get(*f.cache, id);
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{6}); });
    Access::MarkWritten(*f.cache, id, 0x20040, 0x10);
    const auto area = ReadWord(f, 0x20010);
    REQUIRE_FALSE(area.preemtive);
    REQUIRE_FALSE(area.in_flight);
    REQUIRE(f.bytes[0x1003f] == 5);
    REQUIRE(f.bytes[0x10040] == 6);
    REQUIRE(f.bytes[0x1004f] == 6);
    REQUIRE(f.bytes[0x10050] == 5);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Early downloads keep CPU writes, and stop for pages no longer read", "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    // The guest writes part of the page before the early download lands: its values stay.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    std::memset(f.bytes + 0x10020, 0xee, 0x10);
    const std::array<std::pair<DAddr, u64>, 1> applied{{{0x20020, 0x10}}};
    f.cache->ClaimCpuWrites(applied);
    f.runtime.Complete(f.runtime.submitted_tick);
    REQUIRE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x1001f] == 3);
    REQUIRE(f.bytes[0x10020] == 0xee);
    REQUIRE(f.bytes[0x1002f] == 0xee);
    REQUIRE(f.bytes[0x10030] == 3);
    // Not read any more: a few more writes get early downloads, then none.
    const u32 queued = f.runtime.queued_host_reads;
    for (u8 frame = 4; frame < 12; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        f.runtime.Complete(f.runtime.submitted_tick);
    }
    REQUIRE(f.runtime.queued_host_reads - queued == 4);
    const auto area = ReadWord(f, 0x20010);
    REQUIRE_FALSE(area.preemtive);
    REQUIRE(f.bytes[0x10010] == 11);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("GPU writes to guest memory land the early downloads under them first",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    ReadRasterizer reads{*f.cache};
    Tegra::MemoryManager gpu_memory{f.system, f.device, 32, 1ULL << 31, 16};
    gpu_memory.BindRasterizer(&reads);
    gpu_memory.Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    REQUIRE(VideoCore::g_read_downloads == 1);
    // A semaphore release into the copied bytes before the host GPU ran the copy.
    gpu_memory.Write<u32>(0x200040, 0xcafef00d);
    REQUIRE(VideoCore::g_read_downloads == 0);
    u32 value{};
    std::memcpy(&value, f.bytes + 0x10040, sizeof(value));
    REQUIRE(value == 0xcafef00d);
    REQUIRE(f.bytes[0x1003f] == 3);
    REQUIRE(f.bytes[0x10044] == 3);
    REQUIRE(ReadWord(f, 0x20040).preemtive);
}

TEST_CASE("Finished early downloads land a few draws later, or at the next dispatch",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    for (u32 draw = 1; draw < 32; ++draw) {
        f.cache->NoteWorkRecorded(false);
    }
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    f.cache->NoteWorkRecorded(false);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(f.bytes[0x10010] == 3);
    // A draw writes it: the early download goes behind the next dispatch.
    auto& buffer = Access::Get(*f.cache, id);
    f.runtime.pending.push_back([&buffer] { std::fill_n(buffer.bytes.begin(), 0x100, u8{4}); });
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->NoteWorkRecorded(false);
    REQUIRE(f.runtime.queued_host_reads == 1);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(f.runtime.queued_host_reads == 2);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(f.bytes[0x10010] == 4);
    REQUIRE(ReadWord(f, 0x20010).preemtive);
}

TEST_CASE("Early downloads stop at 32 on their way", "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x30000);
    // 33 pages, each read after two writes.
    for (u8 frame = 1; frame <= 2; ++frame) {
        for (DAddr page = 0x20000; page < 0x41000; page += 0x1000) {
            WriteByDispatch(f, id, page, 0x100, frame);
            ReadWord(f, page);
        }
    }
    // Their next writes before the host GPU ran any copy: the 33rd page's read flushes.
    for (DAddr page = 0x20000; page < 0x41000; page += 0x1000) {
        WriteByDispatch(f, id, page, 0x100, 3);
    }
    REQUIRE(f.runtime.queued_host_reads == 32);
    REQUIRE(Access::ReadDownloads(*f.cache) == 32);
    const auto last = ReadWord(f, 0x40010);
    REQUIRE_FALSE(last.preemtive);
    REQUIRE_FALSE(last.in_flight);
    REQUIRE(f.bytes[0x30010] == 3);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x21000));
}

TEST_CASE("Early downloads stop at 1 MiB a frame", "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x1000, frame);
        ReadWord(f, 0x20010);
    }
    // A whole page each, read once the host GPU ran it: 256 make 1 MiB.
    for (u32 write = 0; write < 256; ++write) {
        WriteByDispatch(f, id, 0x20000, 0x1000, static_cast<u8>(write));
        f.runtime.Complete(f.runtime.submitted_tick);
        REQUIRE(ReadWord(f, 0x20010).preemtive);
    }
    REQUIRE(f.runtime.queued_host_reads == 256);
    // The frame's later writes leave their reads to flush.
    WriteByDispatch(f, id, 0x20000, 0x1000, 0xab);
    REQUIRE(f.runtime.queued_host_reads == 256);
    const auto area = ReadWord(f, 0x20010);
    REQUIRE_FALSE(area.preemtive);
    REQUIRE_FALSE(area.in_flight);
    REQUIRE(f.bytes[0x10010] == 0xab);
    REQUIRE(f.bytes[0x10fff] == 0xab);
}

TEST_CASE("Read download check compares what early downloads brought with a synchronous flush",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    const auto& checks = Access::ReadChecks(*f.cache);
    const u64 compared = checks.compared;
    // In guest memory before the read: the read flushes, and compares them as they landed.
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    // The early download's own check and the read's.
    REQUIRE(checks.compared - compared == 0x200);
    // On its way at the read: compared as it lands.
    WriteByDispatch(f, id, 0x20000, 0x100, 4);
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x10010] == 4);
    REQUIRE(checks.compared - compared == 0x400);
    REQUIRE(checks.mismatched == 0);
    REQUIRE(checks.uncovered == 0);
    REQUIRE(checks.rewritten == 0);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x20000, 0x1000));
}

TEST_CASE("Read download check finds early downloaded bytes the GPU changed unrecorded",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    // A GPU write into the landed bytes that nothing records: a synchronous flush at the read
    // would bring it.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 0x10, u8{0x56}); });
    // The guest writes others: newer values, not compared.
    std::memset(f.bytes + 0x10080, 0xee, 0x8);
    ReadWord(f, 0x20010);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0x10);
    REQUIRE(Access::ReadChecks(*f.cache).rewritten == 0x8);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
    // Compared once: a later read finds nothing an early download brought since.
    const u64 compared = Access::ReadChecks(*f.cache).compared;
    REQUIRE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(Access::ReadChecks(*f.cache).compared == compared);
}

namespace {
/// Last frame's values of 16 bytes the guest CPU writes each frame (the GPU's copy), and this
/// frame's: three bytes of each word differ, the last word's top two don't.
constexpr std::array<u8, 16> OldGuestValue{0x68, 0x34, 0x12, 0x3f, 0x8d, 0x11, 0x47, 0xbe,
                                           0x20, 0x55, 0x20, 0x41, 0x73, 0x0c, 0x40, 0x3f};
constexpr std::array<u8, 16> NewGuestValue{0xcd, 0x9a, 0x15, 0x3f, 0x95, 0x02, 0x49, 0xbe,
                                           0x4b, 0x13, 0x22, 0x41, 0x54, 0xe6, 0x40, 0x3f};

/// Guest CPU writes into GPU-written memory, as Core::Memory's HandleRasterizerWrite makes them
/// on any core: collected (GPUDirtyMemoryManager::Collect, in 64-byte blocks), then stored once
/// no download landing under them looked at the collected writes before.
class GuestCpuWrites {
public:
    explicit GuestCpuWrites(Cache& cache_) : cache{cache_} {
        cache.SetPendingCpuWrites([this](const std::function<void(DAddr, size_t)>& callback) {
            {
                std::scoped_lock lock{mutex};
                for (const auto& [address, size] : collected) {
                    callback(address, size);
                }
            }
            if (on_look) {
                on_look(++looks);
            }
        });
    }
    ~GuestCpuWrites() {
        cache.SetPendingCpuWrites({});
    }

    void Write(u8* guest, DAddr address, std::span<const u8> value) {
        const DAddr first = Common::AlignDown(address, 64);
        {
            std::scoped_lock lock{mutex};
            collected.emplace_back(first, Common::AlignUp(address + value.size(), 64) - first);
        }
        VideoCore::DownloadLanding::Await(address, value.size());
        std::memcpy(guest, value.data(), value.size());
    }

    /// GPU::InvalidateGPUCache: the collected writes are applied.
    void Apply() {
        std::vector<std::pair<DAddr, u64>> applied;
        {
            std::scoped_lock lock{mutex};
            applied.swap(collected);
        }
        cache.ClaimCpuWrites(applied);
    }

    /// Called after the buffer cache's n-th look at the collected writes.
    std::function<void(int)> on_look;
    int looks{};

private:
    Cache& cache;
    std::mutex mutex;
    std::vector<std::pair<DAddr, u64>> collected;
};

/// Another core writes NewGuestValue at `address` right after the buffer cache's `at`-th look at
/// the collected writes: the look returns once the write is stored or waits for a landing.
class RacingGuestWrite {
public:
    RacingGuestWrite(Fixture& f, GuestCpuWrites& writes_, DAddr address_, int at)
        : writes{writes_}, guest{f.bytes + (address_ - 0x10000)}, address{address_} {
        writes.looks = 0;
        writes.on_look = [this, at](int look) {
            if (look != at) {
                return;
            }
            const u64 waits = VideoCore::DownloadLanding::Waits();
            go.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (!stored.load() && VideoCore::DownloadLanding::Waits() == waits &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
        };
        thread = std::thread([this] {
            while (!go.load() && !quit.load()) {
                std::this_thread::yield();
            }
            if (go.load()) {
                writes.Write(guest, address, NewGuestValue);
                stored.store(true);
            }
        });
    }
    ~RacingGuestWrite() {
        quit.store(true);
        thread.join();
        writes.on_look = {};
    }

    /// Whether the write was stored, waiting up to 5 s for it.
    bool Stored() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (go.load() && !stored.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        return stored.load();
    }

private:
    GuestCpuWrites& writes;
    u8* guest;
    DAddr address;
    std::atomic<bool> go{};
    std::atomic<bool> stored{};
    std::atomic<bool> quit{};
    std::thread thread;
};

/// The buffer at 0x20000, its first page written by a dispatch (the whole storage binding counts)
/// and read back, with OldGuestValue at 0x20a20 in the GPU's copy and in guest memory.
VideoCommon::BufferId GuestWrittenPage(Fixture& f) {
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    std::ranges::fill(buffer.bytes, 0x12);
    std::ranges::copy(OldGuestValue, buffer.bytes.begin() + 0xa20);
    std::ranges::copy(OldGuestValue, f.bytes + 0x10a20);
    Access::MarkWritten(*f.cache, id, 0x20000, 0x1000);
    return id;
}

bool GuestHolds(const Fixture& f, DAddr address, std::span<const u8> value) {
    return std::equal(value.begin(), value.end(), f.bytes + (address - 0x10000));
}
} // namespace

TEST_CASE("A guest write stored while a read download lands keeps its bytes", "[buffer_ports]") {
    Fixture f;
    GuestWrittenPage(f);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    const u64 waits = VideoCore::DownloadLanding::Waits();
    {
        // Collected right after the landing looked at the collected writes.
        RacingGuestWrite race{f, writes, 0x20a20, 1};
        f.cache->FinishReadDownloads(0x20000, 0x1000);
        REQUIRE(race.Stored());
    }
    REQUIRE(GuestHolds(f, 0x20a20, NewGuestValue));
    REQUIRE(f.bytes[0x10a1f] == 0x12);
    REQUIRE(f.bytes[0x10a30] == 0x12);
    // It waited for the landing to end.
    REQUIRE(VideoCore::DownloadLanding::Waits() > waits);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 1);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
}

TEST_CASE("Read download check leaves out a guest write stored while it compares",
          "[buffer_ports]") {
    Fixture f;
    GuestWrittenPage(f);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    {
        // The download is in guest memory: collected right after the check looked at the
        // collected writes, before it reads guest memory.
        RacingGuestWrite race{f, writes, 0x20a20, 2};
        f.cache->FinishReadDownloads(0x20000, 0x1000);
        REQUIRE(race.Stored());
    }
    REQUIRE(GuestHolds(f, 0x20a20, NewGuestValue));
    REQUIRE(Access::ReadChecks(*f.cache).checked == 1);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
}

TEST_CASE("An early download landed at a draw keeps a guest write stored meanwhile",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x1000, frame);
        ReadWord(f, 0x20010);
    }
    // The next write gets its early download. The host GPU runs it, and the GPU thread lands it
    // at a draw while another core writes into the page.
    WriteByDispatch(f, id, 0x20000, 0x1000, 3);
    std::ranges::copy(OldGuestValue, buffer.bytes.begin() + 0xa20);
    f.runtime.Complete(f.runtime.submitted_tick);
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    {
        RacingGuestWrite race{f, writes, 0x20a20, 1};
        for (u32 draw = 0; draw < 32; ++draw) {
            f.cache->NoteWorkRecorded(false);
        }
        REQUIRE(Access::ReadDownloads(*f.cache) == 0);
        REQUIRE(race.Stored());
    }
    REQUIRE(GuestHolds(f, 0x20a20, NewGuestValue));
    REQUIRE(f.bytes[0x10a1f] == 3);
    REQUIRE(f.bytes[0x10a30] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
}

TEST_CASE("Synchronous downloads keep guest writes collected while they wait for the host GPU",
          "[buffer_ports]") {
    Fixture f;
    const bool mapped = GENERATE(false, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    GuestCpuWrites writes{*f.cache};
    // Collected and stored while the GPU thread waits for the host GPU's copies.
    f.runtime.before_completion = [&] {
        f.runtime.before_completion = {};
        writes.Write(f.bytes + 0x10100, 0x20100, NewGuestValue);
    };
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    const std::array<std::pair<DAddr, u64>, 1> ranges{{{0x20000, 0x2000}}};
    Access::Download(*f.cache, ranges);
    REQUIRE(GuestHolds(f, 0x20100, NewGuestValue));
    REQUIRE(f.bytes[0x100ff] == 0x12);
    REQUIRE(f.bytes[0x10140] == 0x12);
    // The same through DownloadMemory, one buffer at a time.
    writes.Apply();
    f.runtime.before_completion = [&] {
        f.runtime.before_completion = {};
        writes.Write(f.bytes + 0x10200, 0x20200, NewGuestValue);
    };
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    f.cache->DownloadMemory(0x20000, 0x2000);
    REQUIRE(GuestHolds(f, 0x20200, NewGuestValue));
    REQUIRE(f.bytes[0x101ff] == 0x12);
    REQUIRE(f.bytes[0x10240] == 0x12);
}

TEST_CASE("Guest writes racing read downloads keep their bytes", "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    u32 seed = 1234;
    const auto next_delay = [&seed] {
        seed = seed * 1103515245 + 12345;
        return (seed >> 8) % 4000;
    };
    const auto spin = [](u32 iterations) {
        for (volatile u32 i = 0; i < iterations; i = i + 1) {
        }
    };
    int lost = 0;
    for (int iteration = 0; iteration < 2000; ++iteration) {
        // The guest's earlier writes are applied, then a dispatch writes the page: 16 of its bytes
        // hold the GPU's value until the guest writes this round's from another core.
        writes.Apply();
        std::array<u8, 16> gpu_value;
        gpu_value.fill(static_cast<u8>(iteration * 2 + 1));
        std::array<u8, 16> cpu_value;
        cpu_value.fill(static_cast<u8>(iteration * 2 + 2));
        std::ranges::copy(gpu_value, buffer.bytes.begin() + 0xa20);
        Access::MarkWritten(*f.cache, id, 0x20000, 0x1000);
        f.cache->DownloadMemoryForRead(0x20000, 0x1000);
        const u32 writer_delay = next_delay();
        const u32 reader_delay = next_delay();
        std::thread writer{[&] {
            spin(writer_delay);
            writes.Write(f.bytes + 0x10a20, 0x20a20, cpu_value);
        }};
        spin(reader_delay);
        f.cache->FinishReadDownloads(0x20000, 0x1000);
        writer.join();
        lost += GuestHolds(f, 0x20a20, cpu_value) ? 0 : 1;
    }
    REQUIRE(lost == 0);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check leaves out early downloaded bytes the guest wrote since",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    // The guest writes 16 of the landed bytes, as GPU::InvalidateGPUCache applies it, and a draw
    // uploads them. Then it writes the landed value back, which no upload has taken yet.
    const std::array<std::pair<DAddr, u64>, 1> written{{{0x20040, 0x10}}};
    const auto guest_write = [&](u8 value) {
        std::memset(f.bytes + 0x10040, value, 0x10);
        f.cache->ClaimCpuWrites(written);
        f.cache->WriteMemory(0x20040, 0x10);
    };
    guest_write(0xee);
    Access::Upload(*f.cache, id, 0x20000, 0x1000);
    REQUIRE(buffer.bytes[0x40] == 0xee);
    guest_write(3);
    // A synchronous flush at the read leaves them to the guest, as the other landed bytes are
    // what it would bring.
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).compared != 0);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check leaves out a guest write an earlier read's landing kept",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x12);
    Access::CheckEveryRead(*f.cache);
    GuestCpuWrites writes{*f.cache};
    // A read's download, then a read whose bytes it brings: that read's check copy comes alone,
    // in the next submission.
    Access::MarkWritten(*f.cache, id, 0x20000, 0x100);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    f.cache->DownloadMemoryForRead(0x20000, 0x1000);
    REQUIRE(Access::ReadDownloads(*f.cache) == 2);
    REQUIRE(Access::ReadChecks(*f.cache).alone_recorded == 1);
    // The guest writes into those bytes. The first download lands at a fence and keeps the
    // write, collected and not yet applied; then the write is applied, before the second lands.
    writes.Write(f.bytes + 0x10040, 0x20040, NewGuestValue);
    f.runtime.Complete(f.runtime.submitted_tick - 1);
    f.cache->PopAsyncFlushes();
    REQUIRE(Access::ReadDownloads(*f.cache) == 1);
    writes.Apply();
    f.cache->WriteMemory(0x20040, 0x40);
    f.cache->FinishReadDownloads(0x20000, 0x1000);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    REQUIRE(GuestHolds(f, 0x20040, NewGuestValue));
    REQUIRE(f.bytes[0x1003f] == 0x12);
    REQUIRE(f.bytes[0x10080] == 0x12);
    REQUIRE(Access::ReadChecks(*f.cache).checked == 2);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check leaves out early downloaded bytes a query result copy replaced",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    // A query result copied into landed bytes on the GPU (SyncValues), which the buffer cache
    // hears of only as outdating downloads. The query cache writes guest memory after this read,
    // and a synchronous flush at the read leaves guest memory as it is.
    f.runtime.pending.push_back(
        [&buffer] { std::fill_n(buffer.bytes.begin() + 0x40, 8, u8{0x56}); });
    f.cache->OutdateReadDownloads(0x20040, 8);
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(buffer.bytes[0x40] == 0x56);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(f.bytes[0x10048] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).compared != 0);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}

TEST_CASE("Read download check leaves out early downloaded bytes an upload replaced",
          "[buffer_ports]") {
    Fixture f;
    const Nx2TitleScope nx2_title;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    auto& buffer = Access::Get(*f.cache, id);
    Access::CheckEveryRead(*f.cache);
    for (u8 frame = 1; frame <= 2; ++frame) {
        WriteByDispatch(f, id, 0x20000, 0x100, frame);
        ReadWord(f, 0x20010);
    }
    WriteByDispatch(f, id, 0x20000, 0x100, 3);
    f.runtime.Complete(f.runtime.submitted_tick);
    f.cache->NoteWorkRecorded(true);
    REQUIRE(Access::ReadDownloads(*f.cache) == 0);
    // The guest writes 16 of the landed bytes where the buffer cache doesn't hear of it, and a
    // draw uploads them. Then it writes the landed value back, which no upload has taken yet.
    std::memset(f.bytes + 0x10040, 0xee, 0x10);
    Access::MarkCpuModified(*f.cache, 0x20040, 0x10);
    Access::Upload(*f.cache, id, 0x20000, 0x1000);
    REQUIRE(buffer.bytes[0x40] == 0xee);
    std::memset(f.bytes + 0x10040, 3, 0x10);
    // A synchronous flush at the read leaves them to the guest: they are not GPU-modified.
    REQUIRE_FALSE(ReadWord(f, 0x20010).preemtive);
    REQUIRE(f.bytes[0x10040] == 3);
    REQUIRE(f.bytes[0x10010] == 3);
    REQUIRE(Access::ReadChecks(*f.cache).compared != 0);
    REQUIRE(Access::ReadChecks(*f.cache).mismatched == 0);
    REQUIRE(Access::ReadChecks(*f.cache).uncovered == 0);
}
