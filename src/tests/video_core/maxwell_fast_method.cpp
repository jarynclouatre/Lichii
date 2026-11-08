// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/macro.h"
#include "video_core/memory_manager.h"
#include "video_core/native_qmd.h"
#include "video_core/nx2.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace VideoCommon {

struct GpuSplitTestAccess {
    static void Initialize(GpuSplit& split, Core::System& system, u64 position = 0) {
        split.system = &system;
        split.ring = std::make_unique<u64[]>(GpuSplit::RingWords);
        split.write_position = position;
        split.consumed_cache = position;
        split.published_local = position;
        split.publish_every = GpuSplit::RingWords;
    }

    static u64 End(const GpuSplit& split) {
        return split.write_position;
    }

    static void Start(GpuSplit& split, Core::System& system) {
        Initialize(split, system);
        split.active = true;
        split.back_thread = std::jthread([&split](std::stop_token stop) { split.BackLoop(stop); });
    }

    static u64 NearWrap() {
        return GpuSplit::RingWords - 1;
    }

    /// The front waits for the back to reach a position (WaitFor).
    static bool FrontWaits(const GpuSplit& split) {
        return split.wait_target.load() != ~u64{0};
    }

    static u64 Execute(GpuSplit& split, u64 position, u64 available) {
        const size_t index = position & GpuSplit::RingMask;
        const size_t size = split.Execute(
            &split.ring[index], std::min<u64>(available - position, GpuSplit::RingWords - index));
        return position + (size != 0 ? size : GpuSplit::RingWords - index);
    }

    /// The position a read waits for the back to reach (WaitForWrites).
    static u64 WritesTarget(const GpuSplit& split, const Tegra::MemoryManager& memory,
                            GPUVAddr address, u64 size) {
        return split.WritesTarget(memory, address, size);
    }

    /// Whether writes to two device addresses share a slot of the noted write positions.
    static bool ShareWriteSlot(DAddr a, DAddr b) {
        return GpuSplit::WriteSlot(a >> GpuSplit::WriteGranuleBits) ==
               GpuSplit::WriteSlot(b >> GpuSplit::WriteGranuleBits);
    }

    /// Runs what was sent and not run yet on the calling thread, as the back does.
    static void RunBack(GpuSplit& split) {
        GpuSplit::on_back_thread = true;
        u64 position = split.consumed.load();
        while (position < split.write_position) {
            position = Execute(split, position, split.write_position);
        }
        split.consumed.store(position);
        GpuSplit::on_back_thread = false;
    }
};

} // namespace VideoCommon

namespace {

using Tegra::Engines::Maxwell3D;

constexpr u32 CbSize = MAXWELL3D_REG_INDEX(const_buffer.size);
constexpr u32 CbAddressHigh = MAXWELL3D_REG_INDEX(const_buffer.address_high);
constexpr u32 CbAddressLow = MAXWELL3D_REG_INDEX(const_buffer.address_low);
constexpr u32 CbOffset = MAXWELL3D_REG_INDEX(const_buffer.offset);
constexpr u32 CbData = MAXWELL3D_REG_INDEX(const_buffer.buffer);
constexpr u32 DrawBegin = MAXWELL3D_REG_INDEX(draw.begin);
constexpr u32 DrawEnd = MAXWELL3D_REG_INDEX(draw.end);
constexpr u32 ShadowControl = MAXWELL3D_REG_INDEX(shadow_ram_control);

// Executable methods without memory or macro effects a null rasterizer cannot take.
constexpr std::array<u32, 16> executable_methods{
    DrawBegin,
    DrawEnd,
    MAXWELL3D_REG_INDEX(vertex_buffer.first),
    MAXWELL3D_REG_INDEX(vertex_buffer.count),
    MAXWELL3D_REG_INDEX(index_buffer.first),
    MAXWELL3D_REG_INDEX(index_buffer.count),
    MAXWELL3D_REG_INDEX(topology_override),
    MAXWELL3D_REG_INDEX(bind_groups[0].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[1].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[2].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[3].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[4].raw_config),
    MAXWELL3D_REG_INDEX(vertex_array_instance_first),
    MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent),
    Maxwell3D::RootTables::SelectMethod,
    Maxwell3D::RootTables::LoadMethod,
};

struct Fixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Maxwell3D slow{memory};
    Maxwell3D fast{memory};
    std::vector<u32> state_methods;

    Fixture() {
        slow.fast_maxwell3d = false;
        fast.fast_maxwell3d = true;
        for (auto* engine : {&slow, &fast}) {
            engine->BindRasterizer(&rasterizer);
            VideoCommon::Dirty::SetupDirtyFlags(engine->dirty.tables);
            // Give every register a flag so that a missed or extra write shows.
            for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
                if (engine->dirty.tables[1][method] == 0) {
                    engine->dirty.tables[1][method] =
                        static_cast<u8>(VideoCommon::Dirty::LastCommonEntry + method % 61);
                }
            }
            engine->dirty.flags.reset();
            // An unmapped buffer: the upload writes nothing, the offset still advances.
            engine->CallMethod(system, CbAddressHigh, 0x1, true);
            engine->CallMethod(system, CbAddressLow, 0x2000'0000, true);
            engine->CallMethod(system, CbSize, 0x10000, true);
        }
        for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
            if (!slow.execution_mask[method] && (method < CbSize || method > CbAddressLow)) {
                state_methods.push_back(method);
            }
        }
    }

    void Call(u32 method, u32 argument, bool is_last) {
        slow.CallMethod(system, method, argument, is_last);
        fast.CallMethod(system, method, argument, is_last);
    }

    void CallMulti(u32 method, const std::vector<u32>& words) {
        const u32 amount = static_cast<u32>(words.size());
        slow.CallMultiMethod(system, method, words.data(), amount, amount);
        fast.CallMultiMethod(system, method, words.data(), amount, amount);
    }

    void Compare() const {
        REQUIRE(fast.regs.reg_array == slow.regs.reg_array);
        REQUIRE(fast.shadow_state.reg_array == slow.shadow_state.reg_array);
        REQUIRE(fast.shadow_state.shadow_ram_control == slow.shadow_state.shadow_ram_control);
        REQUIRE(fast.dirty.flags == slow.dirty.flags);
        const auto& a = fast.draw_manager.draw_state;
        const auto& b = slow.draw_manager.draw_state;
        REQUIRE(a.topology == b.topology);
        REQUIRE(a.draw_mode == b.draw_mode);
        REQUIRE(a.draw_indexed == b.draw_indexed);
        REQUIRE(a.draw_element_count == b.draw_element_count);
        REQUIRE(a.base_index == b.base_index);
        REQUIRE(a.base_instance == b.base_instance);
        REQUIRE(a.instance_count == b.instance_count);
        REQUIRE(a.drawn_instances == b.drawn_instances);
        REQUIRE(std::memcmp(&a.vertex_buffer, &b.vertex_buffer, sizeof(a.vertex_buffer)) == 0);
        REQUIRE(std::memcmp(&a.index_buffer, &b.index_buffer, sizeof(a.index_buffer)) == 0);
        for (size_t stage = 0; stage < fast.state.shader_stages.size(); ++stage) {
            for (size_t slot = 0; slot < Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& x = fast.state.shader_stages[stage].const_buffers[slot];
                const auto& y = slow.state.shader_stages[stage].const_buffers[slot];
                REQUIRE(x.address == y.address);
                REQUIRE(x.size == y.size);
                REQUIRE(x.enabled == y.enabled);
            }
        }
        REQUIRE(fast.root_tables.tables == slow.root_tables.tables);
        REQUIRE(fast.root_tables.selected == slow.root_tables.selected);
        REQUIRE(fast.root_tables.offset == slow.root_tables.offset);
    }
};

u32 RandomArgument(std::mt19937& rng, u32 method, const Maxwell3D& engine) {
    // Mostly values the register already holds or small ones, as the macros write them.
    switch (rng() % 4) {
    case 0:
        return engine.regs.reg_array[method];
    case 1:
        return rng() % 4;
    default:
        break;
    }
    if (method == DrawBegin) {
        // Instance id in bits 26-27: first, subsequent (an instanced loop) or unchanged.
        return (static_cast<u32>(rng() % 3) << 26) | (rng() % 8);
    }
    if (method == Maxwell3D::RootTables::SelectMethod) {
        return (rng() % 8) | ((rng() % 64 * 4) << 8);
    }
    return rng() % 64;
}

/// A Turing MME instruction from its fields (TuringMacroInterpreter::Decode); a lane left alone
/// adds the zero register to itself.
struct Mme {
    bool end{};
    std::array<u32, 2> op{}, dst{24, 24}, src_a{24, 24}, src_b{24, 24}, imm{};
    std::array<u32, 2> out_method{}, out_emit{};
};

std::array<u32, 3> Encode(const Mme& fields) {
    std::array<u32, 3> bits{};
    const auto field = [&bits](u32 offset, u32 count, u32 value) {
        for (u32 bit = 0; bit < count; ++bit) {
            bits[(offset + bit) / 32] |= ((value >> bit) & 1) << ((offset + bit) % 32);
        }
    };
    field(0, 1, fields.end ? 1 : 0);
    for (u32 lane = 0; lane < 2; ++lane) {
        field(10 + lane * 36, 5, fields.op[lane]);
        field(15 + lane * 36, 5, fields.dst[lane]);
        field(20 + lane * 36, 5, fields.src_a[lane]);
        field(25 + lane * 36, 5, fields.src_b[lane]);
        field(30 + lane * 36, 16, fields.imm[lane]);
        field(82 + lane * 7, 3, fields.out_method[lane]);
        field(85 + lane * 7, 4, fields.out_emit[lane]);
    }
    return {bits[2], bits[1], bits[0]};
}

/// A Turing MME instruction that sends the next macro parameter to `method` (as in
/// native_macro_fifo.cpp); `end` makes the instruction after it the macro's last.
std::array<u32, 3> FifoInstruction(u32 method, bool end) {
    return Encode(Mme{.end = end, .imm = {method, 0}, .out_method = {5, 0}, .out_emit = {3, 0}});
}

} // Anonymous namespace

TEST_CASE("Maxwell3D fast method dispatch matches CallMethod on random method streams",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    for (const u32 seed : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        CAPTURE(seed);
        std::mt19937 rng{seed};
        for (u32 step = 0; step < 20000; ++step) {
            CAPTURE(step);
            const u32 kind = rng() % 100;
            if (kind < 45) {
                const u32 method = f.state_methods[rng() % f.state_methods.size()];
                f.Call(method, RandomArgument(rng, method, f.slow), rng() % 2 == 0);
            } else if (kind < 75) {
                const u32 method = executable_methods[rng() % executable_methods.size()];
                f.Call(method, RandomArgument(rng, method, f.slow), rng() % 2 == 0);
            } else if (kind < 85) {
                f.Call(CbData + rng() % 16, rng(), true);
            } else if (kind < 88) {
                f.Call(CbOffset, (rng() % 0x4000) * 4, true);
            } else if (kind < 92) {
                std::vector<u32> words(1 + rng() % 8);
                for (u32& word : words) {
                    word = rng() % 3 == 0 ? rng() : rng() % 4;
                }
                const u32 method = rng() % 2 == 0
                                       ? CbData + rng() % 16
                                       : f.state_methods[rng() % f.state_methods.size()];
                f.CallMulti(method, words);
            } else if (kind < 94) {
                // Track, track with filter, passthrough or replay.
                f.Call(ShadowControl, rng() % 4, true);
            } else if (kind < 97) {
                for (auto* engine : {&f.slow, &f.fast}) {
                    engine->dirty.flags.reset();
                }
            } else {
                for (auto* engine : {&f.slow, &f.fast}) {
                    engine->draw_manager.DrawDeferred(*engine);
                }
            }
            f.Compare();
        }
    }
}

TEST_CASE("Maxwell3D fast method dispatch issues deferred instanced draws as CallMethod does",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    constexpr u32 subsequent = 1u << 26;
    const auto draw = [&f](u32 begin) {
        f.Call(DrawBegin, begin, true);
        f.Call(MAXWELL3D_REG_INDEX(vertex_buffer.first), 0, true);
        f.Call(MAXWELL3D_REG_INDEX(vertex_buffer.count), 3, true);
        f.Call(DrawEnd, 0, true);
        f.Compare();
    };
    draw(0);
    draw(subsequent);
    draw(subsequent);
    REQUIRE(f.fast.draw_manager.HasDeferredDraw());
    // A constant buffer upload, a bind and state changes each end the loop.
    f.Call(CbData, 7, true);
    f.Compare();
    REQUIRE_FALSE(f.fast.draw_manager.HasDeferredDraw());
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 1 | (3 << 4), true);
    f.Compare();
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(topology_override), 5, true);
    f.Compare();
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(global_base_instance_index), 9, true);
    f.Compare();
    REQUIRE_FALSE(f.fast.draw_manager.HasDeferredDraw());
}

TEST_CASE("Maxwell3D constant buffer word runs match single CallMethod writes",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    std::mt19937 rng{11};
    constexpr u32 subsequent = 1u << 26;
    for (u32 step = 0; step < 20000; ++step) {
        CAPTURE(step);
        switch (rng() % 8) {
        case 0:
            f.Call(ShadowControl, rng() % 4, true);
            break;
        case 1:
            // Near the end too, where words are dropped.
            f.Call(CbSize, rng() % 2 == 0 ? 0x10000 : 0x100 + rng() % 0x40, true);
            break;
        case 2:
            f.Call(CbOffset, rng() % 3 == 0 ? 0xF0 + (rng() % 8) * 4 : (rng() % 0x400) * 4, true);
            break;
        case 3:
            // An instanced loop the first word ends.
            f.Call(DrawBegin, rng() % 2 == 0 ? 0 : subsequent, true);
            f.Call(DrawEnd, 0, true);
            break;
        default: {
            const u32 count = 1 + rng() % 64;
            std::vector<u32> methods(count);
            std::vector<u32> arguments(count);
            const u32 first = CbData + rng() % 16;
            for (u32 i = 0; i < count; ++i) {
                methods[i] = rng() % 2 == 0 ? first : CbData + (first - CbData + i) % 16;
                arguments[i] = rng() % 2 == 0 ? rng() : rng() % 4;
            }
            for (u32 i = 0; i < count; ++i) {
                f.slow.CallMethod(f.system, methods[i], arguments[i], true);
            }
            f.fast.CallConstBufferWords(f.system, methods.data(), arguments.data(), count);
            break;
        }
        }
        f.Compare();
    }
}

TEST_CASE("GpuSplit CB batching respects device remaps within GPU big pages",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    auto* replacement = f.physical.GetPointerFromRaw<u8>(0x30000);
    for (u64 page = 0; page < 16; ++page) {
        table.entries.GetAndFault(4 + page)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing) - 0x4000);
    }
    table.entries.GetAndFault(0x20).Store(false, Common::PageType::Memory, 1,
                                        reinterpret_cast<uintptr_t>(replacement) - 0x20000);
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(0x10000, 0x10000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(0x10000, 0x4000, 0x10000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.device_memory.Unmap(0x11000, 0x1000);
    f.device_memory.Map(0x11000, 0x20000, 0x1000, asid);
    REQUIRE(f.memory.GetPointer(0x100ffc) == backing + 0xffc);
    REQUIRE(f.memory.GetPointer(0x101000) == replacement);

    for (auto* engine : {&f.slow, &f.fast}) {
        std::memset(backing, 0, 0x10000);
        std::memset(replacement, 0, 0x1000);
        VideoCommon::GpuSplit split;
        Access::Initialize(split, f.system);
        split.MacroWrite(engine, CbAddressHigh, 0);
        split.MacroWrite(engine, CbAddressLow, 0x100ffc);
        split.MacroWrite(engine, CbSize, 8);
        split.MacroWrite(engine, CbOffset, 0);
        split.MacroWrite(engine, CbData, 0x12345678);
        split.MacroWrite(engine, CbData + 1, 0x87654321);
        u64 position{};
        while (position < Access::End(split)) {
            position = Access::Execute(split, position, Access::End(split));
        }
        u32 first{}, second{}, stale{};
        std::memcpy(&first, backing + 0xffc, sizeof(first));
        std::memcpy(&second, replacement, sizeof(second));
        std::memcpy(&stale, backing + 0x1000, sizeof(stale));
        CAPTURE(engine == &f.fast);
        CAPTURE(stale);
        REQUIRE(first == 0x12345678);
        REQUIRE(second == 0x87654321);
        REQUIRE(stale == 0);
        REQUIRE(engine->regs.const_buffer.offset == 8);
    }
    f.Compare();
}

TEST_CASE("GpuSplit dispatch preserves mapped uploads and macro boundaries",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    constexpr GPUVAddr gpu_address = 0x100000;
    constexpr DAddr device_address = 0x10000;
    for (u64 page = 0; page < 3; ++page) {
        const VAddr virtual_address = 0x4000 + page * 0x1000;
        auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000 + page * 0x8000);
        table.entries.GetAndFault(virtual_address >> 12)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing) - virtual_address);
    }
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(device_address, 0x3000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(device_address, 0x4000, 0x3000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(gpu_address, device_address, 0x3000, Tegra::PTEKind::PITCH, false);

    struct Snapshot {
        std::array<u8, 0x3000> bytes{};
        std::array<u32, Maxwell3D::Regs::NUM_REGS> regs{};
        std::array<u32, Maxwell3D::Regs::NUM_REGS> shadow{};
        Maxwell3D::DirtyState::Flags dirty{};
        GPUVAddr segment{};
    };
    const auto snapshot = [&](const Maxwell3D& engine) {
        Snapshot result;
        f.memory.ReadBlockUnsafe(gpu_address, result.bytes.data(), result.bytes.size());
        result.regs = engine.regs.reg_array;
        result.shadow = engine.shadow_state.reg_array;
        result.dirty = engine.dirty.flags;
        result.segment = engine.current_dma_segment;
        return result;
    };

    for (const bool wrap : {false, true}) {
        CAPTURE(wrap);
        std::vector<Snapshot> expected;
        for (auto* engine : {&f.slow, &f.fast}) {
            VideoCommon::GpuSplit split;
            struct ProbeEngine final : Tegra::Engines::EngineInterface {
                u32 writes{};
                void CallMethod(Core::System&, u32, u32, bool) override {
                    ++writes;
                }
                void CallMultiMethod(Core::System&, u32, const u32*, u32 amount, u32) override {
                    writes += amount;
                }
            } probe;
            const u64 start = wrap ? Access::NearWrap() : 0;
            Access::Initialize(split, f.system, start);
            std::vector<u64> boundaries;
            std::array<u8, 0x3000> cleared{};
            f.memory.WriteBlockUnsafe(gpu_address, cleared.data(), cleared.size());
            engine->current_dma_segment = 0;
            engine->dirty.flags.reset();
            const auto state = [&](u32 method, u32 value) {
                split.Method(engine, method, value, true, nullptr);
                boundaries.push_back(Access::End(split));
            };
            const auto direct = [&](u32 method, u32 value) {
                split.MacroWrite(engine, method, value);
                boundaries.push_back(Access::End(split));
            };
            size_t observed{};
            const auto inspect = [&] {
                split.Defer([&, engine] {
                    auto actual = snapshot(*engine);
                    if (engine == &f.slow) {
                        expected.push_back(std::move(actual));
                    } else {
                        REQUIRE(observed < expected.size());
                        const auto& reference = expected[observed];
                        REQUIRE(actual.bytes == reference.bytes);
                        REQUIRE(actual.regs == reference.regs);
                        REQUIRE(actual.shadow == reference.shadow);
                        REQUIRE(actual.dirty == reference.dirty);
                        REQUIRE(actual.segment == reference.segment);
                    }
                    ++observed;
                });
                boundaries.push_back(Access::End(split));
            };
            state(ShadowControl, 0);
            state(CbAddressHigh, 0);
            state(CbAddressLow, static_cast<u32>(gpu_address));
            state(CbSize, 0x3000);
            state(CbOffset, 0xffb);
            direct(MAXWELL3D_REG_INDEX(global_base_instance_index), 17);
            inspect();
            for (u32 i = 0; i < 129; ++i) {
                if (i % 2 == 0) {
                    direct(CbData + i % 16, 0x12340000 + i);
                } else {
                    state(CbData + i % 16, 0x56780000 + i);
                }
            }
            inspect();
            state(ShadowControl, 3);
            state(CbOffset, 0x1ffd);
            for (u32 i = 0; i < 33; ++i) {
                direct(CbData + i % 16, 0xdeadbeef);
            }
            inspect();
            state(ShadowControl, 2);
            state(CbOffset, 0x1ffd);
            direct(DrawBegin, 0);
            direct(DrawEnd, 0);
            direct(DrawBegin, 1u << 26);
            direct(DrawEnd, 0);
            direct(CbData, 0x11223344);
            direct(CbData + 1, 0xaabbccdd);
            const GPUVAddr segment = 0x12345678;
            split.Method(engine, CbData + 2, 0x87654321, false, &segment);
            boundaries.push_back(Access::End(split));
            direct(CbData + 3, 0x76543210);
            split.MacroWrite(&probe, 0x10, 42);
            boundaries.push_back(Access::End(split));
            direct(CbData + 4, 0x65432100);
            const std::array<u32, 3> multi{19, 20, 21};
            split.MultiMethod(engine, CbData, multi.data(), 3, 3, &segment);
            boundaries.push_back(Access::End(split));
            inspect();
            state(CbSize, 7);
            state(CbOffset, 0);
            direct(CbData, 0x99aabbcc);
            direct(CbData + 1, 0xddeeff00);
            inspect();

            u64 position = start;
            if (wrap) {
                for (const u64 available : boundaries) {
                    while (position < available) {
                        position = Access::Execute(split, position, available);
                        REQUIRE(position <= available);
                    }
                }
            } else {
                while (position < Access::End(split)) {
                    position = Access::Execute(split, position, Access::End(split));
                    REQUIRE(position <= Access::End(split));
                }
            }
            REQUIRE(observed == expected.size());
            REQUIRE(probe.writes == 1);
        }
        f.Compare();
    }
}

TEST_CASE("GpuSplit CB batching preserves words at the GPU address space end",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    table.entries.GetAndFault(4).Store(false, Common::PageType::Memory, 1,
                                     reinterpret_cast<uintptr_t>(backing) - 0x4000);
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(0x10000, 0x1000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(0x10000, 0x4000, 0x1000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(0xfffff000, 0x10000, 0x1000, Tegra::PTEKind::PITCH, false);
    for (auto* engine : {&f.slow, &f.fast}) {
        std::memset(backing, 0, 0x1000);
        VideoCommon::GpuSplit split;
        Access::Initialize(split, f.system);
        split.MacroWrite(engine, CbAddressHigh, 0);
        split.MacroWrite(engine, CbAddressLow, 0xfffffffcu);
        split.MacroWrite(engine, CbSize, 8);
        split.MacroWrite(engine, CbOffset, 0);
        split.MacroWrite(engine, CbData, 0x12345678);
        split.MacroWrite(engine, CbData + 1, 0x87654321);
        u64 position{};
        while (position < Access::End(split)) {
            position = Access::Execute(split, position, Access::End(split));
        }
        u32 word{};
        std::memcpy(&word, backing + 0xffc, sizeof(word));
        CAPTURE(engine == &f.fast);
        REQUIRE(word == 0x12345678);
        REQUIRE(engine->regs.const_buffer.offset == 8);
    }
    f.Compare();
}

TEST_CASE("Maxwell3D fast method dispatch cost", "[.][maxwell_fast_method_bench]") {
    Fixture f;
    // Per draw, as the back thread sees Ocarina's: state writes the front did not drop, a
    // constant buffer update, three binds and a draw.
    std::vector<std::pair<u32, u32>> stream;
    std::mt19937 rng{1};
    for (u32 draw = 0; draw < 4096; ++draw) {
        for (u32 i = 0; i < 12; ++i) {
            stream.emplace_back(f.state_methods[rng() % 512], rng() % 8);
        }
        stream.emplace_back(CbOffset, 0x640);
        stream.emplace_back(CbData, draw);
        stream.emplace_back(CbData + 1, 0);
        stream.emplace_back(CbData + 2, draw * 3);
        for (const u32 stage : {0u, 2u, 4u}) {
            stream.emplace_back(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config) + stage * 8,
                                1 | (rng() % 8) << 4);
        }
        stream.emplace_back(DrawBegin, 0);
        stream.emplace_back(MAXWELL3D_REG_INDEX(vertex_buffer.first), rng() % 4);
        stream.emplace_back(MAXWELL3D_REG_INDEX(vertex_buffer.count), 3 + rng() % 4);
        stream.emplace_back(DrawEnd, 0);
    }
    const auto run = [&](Maxwell3D& engine) {
        const auto start = std::chrono::steady_clock::now();
        for (u32 round = 0; round < 50; ++round) {
            for (const auto& [method, argument] : stream) {
                engine.CallMethod(f.system, method, argument, true);
            }
        }
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                   .count() /
               (50.0 * static_cast<double>(stream.size()));
    };
    run(f.slow);
    run(f.fast);
    const double slow_ns = run(f.slow);
    const double fast_ns = run(f.fast);
    WARN("CallMethod: " << slow_ns << " ns per method, fast path " << fast_ns << " ns");

    // Three constant buffer words (a draw macro's c[0][0x640..0x648]) one by one or as a run.
    constexpr std::array<u32, 3> methods{CbData, CbData + 1, CbData + 2};
    constexpr std::array<u32, 3> arguments{1, 2, 3};
    const auto words = [&](bool as_run) {
        const auto start = std::chrono::steady_clock::now();
        for (u32 i = 0; i < 1'000'000; ++i) {
            f.fast.CallMethod(f.system, CbOffset, 0x640, true);
            if (as_run) {
                f.fast.CallConstBufferWords(f.system, methods.data(), arguments.data(), 3);
            } else {
                for (u32 w = 0; w < 3; ++w) {
                    f.fast.CallMethod(f.system, methods[w], arguments[w], true);
                }
            }
        }
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                   .count() /
               1'000'000.0;
    };
    words(false);
    words(true);
    const double single_ns = words(false);
    const double run_ns = words(true);
    WARN("Three constant buffer words: " << single_ns << " ns one by one, " << run_ns
                                         << " ns as a run (unmapped buffer)");
}

TEST_CASE("Full split drains preserve aliased uploads, partial coverage and queued remaps",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    constexpr VAddr cpu_address = 0x4000;
    constexpr DAddr device_address = 0x10000;
    constexpr GPUVAddr gpu_address = 0x100000;
    constexpr GPUVAddr alias = 0x200000;
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    for (u64 page = 0; page < 2; ++page) {
        const VAddr address = cpu_address + page * 0x1000;
        table.entries.GetAndFault(address >> 12)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing + page * 0x1000) - address);
    }
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    f.device_memory.Map(device_address, cpu_address, 0x2000, asid);
    SCOPE_EXIT {
        f.device_memory.Unmap(device_address, 0x2000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(gpu_address, device_address, 0x2000, Tegra::PTEKind::PITCH, false);
    f.memory.Map(alias, device_address, 0x1000, Tegra::PTEKind::PITCH, false);
    std::memset(backing, 0, 0x2000);

    VideoCommon::GpuSplit split;
    Access::Start(split, f.system);
    SCOPE_EXIT { split.Stop(); };
    const auto state = [&](u32 method, u32 value) {
        split.Method(&f.fast, method, value, true, nullptr);
    };
    const auto upload = [&](GPUVAddr address, u32 size, std::vector<u32> words) {
        state(ShadowControl, 0);
        state(CbAddressHigh, static_cast<u32>(address >> 32));
        state(CbAddressLow, static_cast<u32>(address));
        state(CbSize, size);
        state(CbOffset, 0);
        split.MultiMethod(&f.fast, CbData, words.data(), static_cast<u32>(words.size()),
                          static_cast<u32>(words.size()), nullptr);
        std::ranges::fill(words, 0xdeadbeef);
    };

    upload(gpu_address, 16, {1, 2, 3, 4});
    upload(alias + 4, 4, {20, 30});
    upload(gpu_address + 0x1000, 8, {50, 60});
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    std::array<u32, 4> read{};
    f.memory.ReadBlockUnsafe(gpu_address, read.data(), sizeof(read));
    REQUIRE(read == std::array<u32, 4>{1, 20, 3, 4});
    REQUIRE(f.memory.Read<u32>(alias + 4) == 20);
    REQUIRE(f.memory.Read<u32>(gpu_address + 0x1000) == 50);

    split.Defer([&] {
        f.memory.Map(alias, device_address + 0x1000, 0x1000, Tegra::PTEKind::PITCH, false);
    });
    upload(alias, 4, {70});
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    REQUIRE(f.memory.Read<u32>(gpu_address) == 1);
    REQUIRE(f.memory.Read<u32>(gpu_address + 0x1000) == 70);
    REQUIRE(f.memory.Read<u32>(alias + 4) == 60);
}

TEST_CASE("Split consumption does not complete deferred downstream callbacks",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    Access::Start(split, f.system);
    SCOPE_EXIT { split.Stop(); };
    std::vector<std::function<void()>> deferred;
    std::array<u32, 3> output{};
    for (u32 i = 0; i < output.size(); ++i) {
        split.Defer([&, i] { deferred.emplace_back([&, i] { output[i] = i + 1; }); });
    }
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    REQUIRE(deferred.size() == 3);
    REQUIRE(output == std::array<u32, 3>{});
    split.Run([&] {
        for (auto& operation : deferred) {
            operation();
        }
        deferred.clear();
    });
    REQUIRE(output == std::array<u32, 3>{1, 2, 3});
}

TEST_CASE("Split back runs a requested tick while it waits for work and between entries",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    std::atomic<u32> ticks{};
    std::atomic<bool> entered{};
    std::atomic<bool> release{};
    u32 ticks_at_second{~0U};
    split.SetTickHook([](void* counter) { static_cast<std::atomic<u32>*>(counter)->fetch_add(1); },
                      &ticks);
    // Before the back runs, a request does nothing, nor is it kept for later.
    split.RequestTick();
    Access::Start(split, f.system);
    SCOPE_EXIT {
        split.Stop();
    };
    // Runs before Stop: a failed check must not leave the back in the entry below.
    SCOPE_EXIT {
        release = true;
    };
    const auto wait_until = [](const auto& done) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!done() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        return done();
    };
    REQUIRE(wait_until([&] { return split.BackRunning(); }));
    // The back has nothing to do and goes to sleep: a request wakes it and runs at once.
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    REQUIRE(ticks.load() == 0);
    split.RequestTick();
    REQUIRE(wait_until([&] { return ticks.load() == 1; }));
    // The back runs an entry: the request runs once it ends, before the next entry.
    split.Defer([&] {
        entered = true;
        while (!release.load()) {
            std::this_thread::yield();
        }
    });
    split.Defer([&] { ticks_at_second = ticks.load(); });
    split.Publish();
    REQUIRE(wait_until([&] { return entered.load(); }));
    split.RequestTick();
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    REQUIRE(ticks.load() == 1);
    release = true;
    split.Drain(VideoCommon::DrainSite::Other);
    REQUIRE(ticks_at_second == 2);
    REQUIRE(ticks.load() == 2);
}

TEST_CASE("Split command list fetches wait only for the writers before the last host sync",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    std::array<std::atomic<bool>, 3> release{};
    std::atomic<bool> done{};
    bool waited{};
    std::thread waiter;
    Access::Start(split, f.system);
    SCOPE_EXIT {
        split.Stop();
    };
    // Runs before Stop: a failed check must not leave the back in an entry or the waiter waiting.
    SCOPE_EXIT {
        for (auto& flag : release) {
            flag = true;
        }
        if (waiter.joinable()) {
            waiter.join();
        }
    };
    // An entry the back stays in until its flag is set.
    const auto blocking = [&split](std::atomic<bool>& flag) {
        split.Defer([&flag] {
            while (!flag.load()) {
                std::this_thread::yield();
            }
        });
    };
    const auto finishes = [&done] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!done.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        return done.load();
    };

    // A writer sent after the last host-level wait: the fetch does not wait for it.
    split.NoteSync();
    blocking(release[0]);
    split.NoteWriter();
    split.Publish();
    REQUIRE_FALSE(split.WaitForSyncedWriters());

    // A wait after it orders that writer before the fetch, but not the writer sent after the wait.
    split.NoteSync();
    blocking(release[1]);
    split.NoteWriter();
    split.Publish();
    waiter = std::thread([&] {
        waited = split.WaitForSyncedWriters();
        done = true;
    });
    // The back reports its position at the target only if the front waits before it passes it.
    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!Access::FrontWaits(split) && !done.load() &&
           std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    REQUIRE_FALSE(done.load());
    release[0] = true;
    REQUIRE(finishes());
    waiter.join();
    REQUIRE(waited);
    release[1] = true;
    split.Drain(VideoCommon::DrainSite::Other);

    // Other work before the wait is not a writer: the fetch does not wait for it either.
    split.Defer([] {});
    split.NoteWriter();
    split.Drain(VideoCommon::DrainSite::Other);
    blocking(release[2]);
    split.NoteSync();
    split.Publish();
    REQUIRE_FALSE(split.WaitForSyncedWriters());
    release[2] = true;
    split.Drain(VideoCommon::DrainSite::Other);
}

TEST_CASE("Split tells whether compute or copy work it was sent is still queued",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    std::atomic<bool> release{};
    Access::Start(split, f.system);
    SCOPE_EXIT {
        split.Stop();
    };
    // Runs before Stop: a failed check must not leave the back in the entry below.
    SCOPE_EXIT {
        release = true;
    };
    REQUIRE_FALSE(split.WritersPending());
    split.Defer([&release] {
        while (!release.load()) {
            std::this_thread::yield();
        }
    });
    split.NoteWriter();
    split.Publish();
    REQUIRE(split.WritersPending());
    release = true;
    split.Drain(VideoCommon::DrainSite::Other);
    REQUIRE_FALSE(split.WritersPending());
    // Other work does not count.
    split.Defer([] {});
    split.Publish();
    REQUIRE_FALSE(split.WritersPending());
    split.Drain(VideoCommon::DrainSite::Other);
}

TEST_CASE("Split remembers the semaphore writes its back has not run yet",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    std::atomic<bool> release{};
    Access::Start(split, f.system);
    SCOPE_EXIT {
        split.Stop();
    };
    // Runs before Stop: a failed check must not leave the back in the entry below.
    SCOPE_EXIT {
        release = true;
    };
    // Releases behind an entry the back is still in.
    split.Defer([&release] {
        while (!release.load()) {
            std::this_thread::yield();
        }
    });
    split.Defer([] {});
    split.NoteSemaphoreWrite(0x1000, 5, true);
    split.Defer([] {});
    split.NoteSemaphoreWrite(0x2002, 7, false);
    split.Publish();
    auto queued = split.QueuedSemaphoreWriteAt(0x1000);
    REQUIRE(queued.has_value());
    REQUIRE(queued->known);
    REQUIRE(queued->payload == 5);
    // By 4-byte word: the second write is at 0x2000. No write was sent to 0x1004.
    queued = split.QueuedSemaphoreWriteAt(0x2000);
    REQUIRE(queued.has_value());
    REQUIRE_FALSE(queued->known);
    REQUIRE_FALSE(split.QueuedSemaphoreWriteAt(0x1004).has_value());
    // The last write sent to a word decides.
    split.Defer([] {});
    split.NoteSemaphoreWrite(0x1000, 9, true);
    split.Publish();
    queued = split.QueuedSemaphoreWriteAt(0x1000);
    REQUIRE(queued.has_value());
    REQUIRE(queued->payload == 9);
    // Once the back ran them, the words hold their values in memory.
    release = true;
    split.Drain(VideoCommon::DrainSite::Other);
    REQUIRE_FALSE(split.QueuedSemaphoreWriteAt(0x1000).has_value());
    REQUIRE_FALSE(split.QueuedSemaphoreWriteAt(0x2000).has_value());
}

TEST_CASE("Split counts a fenced semaphore write in memory once its fence ran, not before",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    using Landing = VideoCommon::GpuSplit::Landing;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    Access::Start(split, f.system);
    SCOPE_EXIT {
        split.Stop();
    };
    // A fence-flagged report lands once the fence after it ran (QueryCacheBase::CounterReport),
    // which tells with the entry the back gave it.
    split.Defer([] {});
    std::optional<u64> entry;
    split.Defer([&split, &entry] { entry = split.BackEntry(); });
    split.NoteSemaphoreWrite(0x1000, 21, true, false, Landing::Fenced);
    split.Drain(VideoCommon::DrainSite::Other);
    REQUIRE(entry.has_value());
    REQUIRE(*entry > 0);
    REQUIRE_FALSE(split.BackEntry().has_value());
    auto queued = split.QueuedSemaphoreWriteAt(0x1000);
    REQUIRE(queued.has_value());
    REQUIRE(queued->fenced);
    REQUIRE(queued->payload == 21);
    // A write that lands as the back runs it, sent before that fence ran: the word holds either.
    split.Defer([] {});
    split.NoteSemaphoreWrite(0x1000, 22, true);
    split.Drain(VideoCommon::DrainSite::Other);
    queued = split.QueuedSemaphoreWriteAt(0x1000);
    REQUIRE(queued.has_value());
    REQUIRE(queued->fenced);
    REQUIRE_FALSE(queued->known);
    // A fence of an entry before it lands nothing of it; its own lands it.
    split.NoteFencedWrite(*entry - 1);
    REQUIRE(split.QueuedSemaphoreWriteAt(0x1000).has_value());
    split.NoteFencedWrite(*entry);
    REQUIRE_FALSE(split.QueuedSemaphoreWriteAt(0x1000).has_value());
    // Once it landed, a write after it holds the word once the back ran it.
    split.Defer([] {});
    split.NoteSemaphoreWrite(0x1000, 23, true);
    queued = split.QueuedSemaphoreWriteAt(0x1000);
    REQUIRE(queued.has_value());
    REQUIRE_FALSE(queued->fenced);
    REQUIRE(queued->payload == 23);
    split.Drain(VideoCommon::DrainSite::Other);
    REQUIRE_FALSE(split.QueuedSemaphoreWriteAt(0x1000).has_value());
    // A later entry's fence lands the fenced writes of the entries before it.
    split.Defer([] {});
    split.NoteSemaphoreWrite(0x2000, 24, true, false, Landing::Fenced);
    std::optional<u64> later;
    split.Defer([&split, &later] { later = split.BackEntry(); });
    split.Drain(VideoCommon::DrainSite::Other);
    REQUIRE(split.QueuedSemaphoreWriteAt(0x2000).has_value());
    REQUIRE(later.has_value());
    split.NoteFencedWrite(*later);
    REQUIRE_FALSE(split.QueuedSemaphoreWriteAt(0x2000).has_value());
}

TEST_CASE("Split waits for a semaphore write's position, not for the work sent after it",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    std::array<std::atomic<bool>, 2> release{};
    Access::Start(split, f.system);
    SCOPE_EXIT {
        split.Stop();
    };
    // Runs before Stop: a failed check must not leave the back in the entries below.
    SCOPE_EXIT {
        release[0] = true;
        release[1] = true;
    };
    const auto blocking = [&split](std::atomic<bool>& flag) {
        split.Defer([&flag] {
            while (!flag.load()) {
                std::this_thread::yield();
            }
        });
    };
    // A release the front can't value (a reduction, a counter) sent by compute work, behind an
    // entry the back is still in, then more work behind another.
    blocking(release[0]);
    split.Defer([] {});
    split.NoteSemaphoreWrite(0x1000, 0, false);
    split.NoteWriter();
    const u64 release_position = split.Position();
    REQUIRE(split.LastWriterPosition() == release_position);
    blocking(release[1]);
    split.Publish();
    const auto queued = split.QueuedSemaphoreWriteAt(0x1000);
    REQUIRE(queued.has_value());
    REQUIRE(queued->position == release_position);
    REQUIRE_FALSE(split.Ran(release_position));
    std::atomic<bool> done{};
    bool waited{};
    std::thread waiter([&] {
        waited = split.WaitForPosition(release_position, VideoCommon::DrainSite::Acquire);
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    REQUIRE_FALSE(done.load());
    release[0] = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!done.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    REQUIRE(done.load());
    waiter.join();
    REQUIRE(waited);
    // The release ran; the work after it still runs.
    REQUIRE(split.Ran(release_position));
    REQUIRE_FALSE(split.QueuedSemaphoreWriteAt(0x1000).has_value());
    REQUIRE_FALSE(split.Ran(split.Position()));
    release[1] = true;
    split.Drain(VideoCommon::DrainSite::Other);
    REQUIRE(split.Ran(split.Position()));
    REQUIRE_FALSE(split.WaitForPosition(release_position, VideoCommon::DrainSite::Acquire));
}

TEST_CASE("Split restart resets writer positions without replaying old releases",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    u32 released{};
    u32 release_count{};
    for (u32 session = 1; session <= 3; ++session) {
        Access::Start(split, f.system);
        SCOPE_EXIT { split.Stop(); };
        REQUIRE(Access::End(split) == 0);
        REQUIRE_FALSE(split.WaitForWriters());
        REQUIRE_FALSE(split.WaitForSync());
        split.Defer([&, session] {
            released = session;
            ++release_count;
        });
        split.NoteWriter();
        split.NoteSync();
        split.Drain(VideoCommon::DrainSite::Acquire);
        REQUIRE(released == session);
        REQUIRE(release_count == session);
        split.Stop();
        REQUIRE_FALSE(split.Active());
        REQUIRE_FALSE(split.BackRunning());
        REQUIRE(Access::End(split) == 0);
    }
}

TEST_CASE("Maxwell3D takes its macros back from the split's front once the split stops",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    auto& engine = f.fast;
    constexpr u32 StartAddressPointer = MAXWELL3D_REG_INDEX(load_mme.start_address_ptr);
    constexpr u32 StartAddress = MAXWELL3D_REG_INDEX(load_mme.start_address);
    const u32 state = f.state_methods.front();
    // The front takes the macros over and tracks a state write in its shadow RAM copy.
    engine.FrontCallMethod(f.system, split, state, 0x1234, true);
    REQUIRE(engine.shadow_state.reg_array[state] != 0x1234);
    // The split stopped: the next engine method runs on this thread, not the split's back.
    engine.CallMethod(f.system, StartAddressPointer, 0, true);
    engine.CallMethod(f.system, StartAddress, 0x40, true);
    CHECK(engine.shadow_state.reg_array[state] == 0x1234);
    CHECK(engine.GetNativeMacroPositions()[0] == 0x40);
}

TEST_CASE("Maxwell3D keeps state written after the split stops over the front's shadow RAM",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    auto& engine = f.fast;
    const u32 state = f.state_methods.front();
    engine.FrontCallMethod(f.system, split, state, 0x1234, true);
    // After the stop, DmaPusher sinks a state write and consumes the sink before the next
    // executable method: the handback must come first so that write wins.
    engine.method_sink.emplace_back(state, 0x5678);
    engine.ConsumeSink(f.system);
    engine.CallMethod(f.system, MAXWELL3D_REG_INDEX(load_mme.start_address_ptr), 0, true);
    CHECK(engine.shadow_state.reg_array[state] == 0x5678);
    CHECK(engine.regs.reg_array[state] == 0x5678);
}

TEST_CASE("Maxwell3D's split front uploads macro code sent as one multi-method run",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    auto& engine = f.fast;
    engine.SetNativeTuringMacroMode(true);
    // GCC 13's loop unswitching dropped these words from the front's per-word loop (#500).
    constexpr u32 pointer = 0x262;
    constexpr std::array<u32, 6> code{0x740003, 0x18c00000, 0x318c0001,
                                      0x740003, 0x18c00000, 0x318c0000};
    engine.FrontCallMethod(f.system, split, MAXWELL3D_REG_INDEX(load_mme.instruction_ptr), pointer,
                           true);
    engine.FrontCallMultiMethod(f.system, split, MAXWELL3D_REG_INDEX(load_mme.instruction),
                                code.data(), code.size(), code.size());
    const auto& macros = engine.GetNativeMacroEngine();
    REQUIRE(macros.native_write_offset == pointer * 3 + code.size());
    CHECK(std::equal(code.begin(), code.end(), macros.native_code.begin() + pointer * 3));
}

TEST_CASE("Split macro DMA reads wait only for the queued work that can write the words, or run "
          "on the back",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    using Writes = VideoCommon::GpuSplit::Writes;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT {
        f.system.Kernel().UnregisterKernelObject(&process);
    };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    constexpr VAddr cpu_address = 0x4000;
    constexpr DAddr device_address = 0x10000;
    constexpr GPUVAddr gpu_address = 0x100000;
    constexpr GPUVAddr alias = 0x200000; // The first page again
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    for (u64 page = 0; page < 2; ++page) {
        const VAddr address = cpu_address + page * 0x1000;
        table.entries.GetAndFault(address >> 12)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing + page * 0x1000) - address);
    }
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    f.device_memory.Map(device_address, cpu_address, 0x2000, asid);
    SCOPE_EXIT {
        f.device_memory.Unmap(device_address, 0x2000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(gpu_address, device_address, 0x2000, Tegra::PTEKind::PITCH, false);
    f.memory.Map(alias, device_address, 0x1000, Tegra::PTEKind::PITCH, false);
    std::memset(backing, 0, 0x2000);
    // The words read below, in two 256-byte granules that do not share a write slot.
    constexpr GPUVAddr word = gpu_address + 0x40;
    constexpr GPUVAddr other = gpu_address + 0x140;
    REQUIRE_FALSE(Access::ShareWriteSlot(device_address + 0x40, device_address + 0x140));

    {
        // Where reads wait: writes are placed by device memory, any GPU address of it.
        VideoCommon::GpuSplit split;
        Access::Initialize(split, f.system);
        split.Defer([] {}, Writes::Noted);
        split.NoteWrite(f.memory, alias + 0x48, 4);
        const u64 noted = Access::End(split);
        split.Defer([] {}, Writes::Noted);
        split.NoteWrite(f.memory, 0x300000, 16); // Unmapped: nothing is written
        CHECK(Access::WritesTarget(split, f.memory, word, 0x18) == noted);
        CHECK(Access::WritesTarget(split, f.memory, other, 0x18) == 0);
        // Writes too long to place, and work nobody noted the writes of, count as anywhere.
        split.Defer([] {}, Writes::Noted);
        split.NoteWrite(f.memory, gpu_address + 0x1000, 0x8000);
        CHECK(Access::WritesTarget(split, f.memory, other, 0x18) == Access::End(split));
        split.Defer([] {});
        CHECK(Access::WritesTarget(split, f.memory, other, 0x18) == Access::End(split));
    }

    // The 3D engine's front, against a running back: macro 0 reads one word with
    // MME_DMA_READ_FIFOED at the address in its first two parameters and sends it to a scratch
    // register (the load after the read takes the word). It also clears a data RAM word, which
    // keeps it on the front. Macro 1 is the same without that and runs on the back; macro 2 (on
    // the front) copies the scratch register to the next one.
    auto& engine = f.fast;
    engine.SetNativeTuringMacroMode(true);
    constexpr u32 state = MAXWELL3D_REG_INDEX(shadow_scratch) + 1;
    constexpr u32 copy = state + 1;
    VideoCommon::GpuSplit split;
    Access::Start(split, f.system);
    SCOPE_EXIT {
        split.Stop();
    };
    std::array<std::atomic<bool>, 5> release{};
    std::atomic<bool> done{};
    std::thread reader;
    SCOPE_EXIT {
        for (auto& flag : release) {
            flag = true;
        }
        if (reader.joinable()) {
            reader.join();
        }
    };
    const auto front = [&](u32 method, u32 argument) {
        engine.FrontCallMethod(f.system, split, method, argument, true);
    };
    std::vector<u32> code;
    const Mme clear_data_ram{
        .op = {0, 31}, .imm = {state, 0}, .out_method = {5, 0}, .out_emit = {3, 0}};
    // Macro 2: the value of `state` (MME register read) sent to `copy`, then a no-op.
    const Mme read_state{.end = true,
                         .op = {22, 0},
                         .src_a = {25, 24},
                         .imm = {state, copy},
                         .out_method = {6, 0},
                         .out_emit = {1, 0}};
    for (const auto instruction :
         {FifoInstruction(0x154, false), FifoInstruction(0x155, false),
          FifoInstruction(0x158, true), Encode(clear_data_ram), FifoInstruction(0x154, false),
          FifoInstruction(0x155, false), FifoInstruction(0x158, true),
          FifoInstruction(state, false), Encode(read_state), Encode(Mme{})}) {
        code.insert(code.end(), instruction.begin(), instruction.end());
    }
    front(MAXWELL3D_REG_INDEX(load_mme.instruction_ptr), 0);
    engine.FrontCallMultiMethod(f.system, split, MAXWELL3D_REG_INDEX(load_mme.instruction),
                                code.data(), static_cast<u32>(code.size()),
                                static_cast<u32>(code.size()));
    front(MAXWELL3D_REG_INDEX(load_mme.start_address_ptr), 0);
    for (const u32 start : {0U, 4U, 8U}) {
        front(MAXWELL3D_REG_INDEX(load_mme.start_address), start);
    }
    const auto dma_read = [&](GPUVAddr address, u32 macro = 0xe00) {
        engine.FrontCallMethod(f.system, split, macro, static_cast<u32>(address >> 32), false);
        engine.FrontCallMethod(f.system, split, macro + 1, static_cast<u32>(address), false);
        engine.FrontCallMethod(f.system, split, macro + 1, 1, true);
    };
    const auto read_on_reader = [&](GPUVAddr address, u32 macro = 0xe00) {
        done = false;
        reader = std::thread([&, address, macro] {
            dma_read(address, macro);
            done = true;
        });
    };
    const auto copy_on_reader = [&] {
        done = false;
        reader = std::thread([&] {
            front(0xe04, 0);
            done = true;
        });
    };
    const auto finishes = [&done] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!done.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        return done.load();
    };
    const auto waits = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!Access::FrontWaits(split) && !done.load() &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        return !done.load();
    };
    // A constant buffer word, sent through the front.
    const auto upload = [&](GPUVAddr address, u32 value) {
        front(CbAddressHigh, static_cast<u32>(address >> 32));
        front(CbAddressLow, static_cast<u32>(address));
        front(CbSize, 0x100);
        front(CbOffset, 0);
        front(CbData, value);
    };
    // An entry the back stays in until its flag is set.
    const auto blocking = [&split](std::atomic<bool>& flag, Writes writes) {
        split.Defer(
            [&flag] {
                while (!flag.load()) {
                    std::this_thread::yield();
                }
            },
            writes);
        split.Publish();
    };

    // The first macro also ends the front's assumption that a draw may be pending.
    upload(word, 0x11111111);
    upload(other, 0x22222222);
    dma_read(word);
    split.Drain(VideoCommon::DrainSite::Other);
    REQUIRE(engine.regs.reg_array[state] == 0x11111111);

    // Work that writes nothing and a write of other words, still queued: the read goes ahead.
    blocking(release[0], Writes::Noted);
    upload(other, 0x33333333);
    read_on_reader(word);
    REQUIRE(finishes());
    reader.join();
    release[0] = true;
    split.Drain(VideoCommon::DrainSite::Other);
    CHECK(engine.regs.reg_array[state] == 0x11111111);
    CHECK(f.memory.Read<u32>(other) == 0x33333333);

    // A write of the word through its other GPU address: the read waits for it and sees it.
    blocking(release[1], Writes::Noted);
    upload(alias + 0x40, 0x44444444);
    read_on_reader(word);
    REQUIRE(waits());
    release[1] = true;
    REQUIRE(finishes());
    reader.join();
    split.Drain(VideoCommon::DrainSite::Other);
    CHECK(engine.regs.reg_array[state] == 0x44444444);

    // Work whose writes are not noted: the read waits for it.
    blocking(release[2], Writes::Anywhere);
    read_on_reader(other);
    REQUIRE(waits());
    release[2] = true;
    REQUIRE(finishes());
    reader.join();
    split.Drain(VideoCommon::DrainSite::Other);
    CHECK(engine.regs.reg_array[state] == 0x33333333);

    // Macro 1 runs on the back: the front goes on past that work, and the back reads the word
    // after the upload queued behind it.
    front(state, 0x77777777);
    blocking(release[3], Writes::Anywhere);
    upload(word, 0x55555555);
    read_on_reader(word, 0xe02);
    REQUIRE(finishes());
    reader.join();
    release[3] = true;
    split.Drain(VideoCommon::DrainSite::Other);
    CHECK(engine.regs.reg_array[state] == 0x55555555);

    // The front's copy of the register is stale: a write of the value it holds is still sent
    // (the back sinks it until the next macro).
    front(state, 0x77777777);
    front(0xe04, 0);
    split.Drain(VideoCommon::DrainSite::Other);
    CHECK(engine.regs.reg_array[state] == 0x77777777);

    // A front macro that reads the register waits for the back's macro and gets its value.
    blocking(release[4], Writes::Noted);
    upload(word, 0x66666666);
    dma_read(word, 0xe02);
    copy_on_reader();
    REQUIRE(waits());
    release[4] = true;
    REQUIRE(finishes());
    reader.join();
    split.Drain(VideoCommon::DrainSite::Other);
    CHECK(engine.regs.reg_array[copy] == 0x66666666);

    // A replay takes the shadow RAM entry the back's macro tracked from the back.
    using Control = Maxwell3D::Regs::ShadowRamControl;
    front(state, 0x99999999);
    upload(word, 0x88888888);
    dma_read(word, 0xe02);
    front(ShadowControl, static_cast<u32>(Control::Replay));
    front(state, 0);
    front(ShadowControl, static_cast<u32>(Control::Track));
    front(0xe04, 0);
    split.Drain(VideoCommon::DrainSite::Other);
    CHECK(engine.regs.reg_array[state] == 0x88888888);
    CHECK(engine.regs.reg_array[copy] == 0x88888888);
}

namespace {

// Two pages of guest memory at gpu_address (device_address) for the tests below.
struct MappedPages {
    static constexpr VAddr cpu_address = 0x4000;
    static constexpr DAddr device_address = 0x10000;
    static constexpr GPUVAddr gpu_address = 0x100000;

    explicit MappedPages(Fixture& f) : fixture{f}, process{f.system.Kernel()} {
        auto& table = process.GetPageTable().GetImpl();
        table.Resize(20, 12);
        backing = f.physical.GetPointerFromRaw<u8>(0x10000);
        for (u64 page = 0; page < 2; ++page) {
            const VAddr address = cpu_address + page * 0x1000;
            table.entries.GetAndFault(address >> 12)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(backing + page * 0x1000) - address);
        }
        process.GetMemory().SetCurrentPageTable(process);
        asid = f.device_memory.RegisterProcess(&process.GetMemory());
        f.device_memory.BindInterface(&f.rasterizer);
        f.device_memory.Map(device_address, cpu_address, 0x2000, asid);
        f.memory.BindRasterizer(&f.rasterizer);
        f.memory.Map(gpu_address, device_address, 0x2000, Tegra::PTEKind::PITCH, false);
        std::memset(backing, 0, 0x2000);
    }
    ~MappedPages() {
        fixture.device_memory.Unmap(device_address, 0x2000);
        fixture.device_memory.UnregisterProcess(asid);
        fixture.system.Kernel().UnregisterKernelObject(&process);
    }
    MappedPages(const MappedPages&) = delete;
    MappedPages& operator=(const MappedPages&) = delete;

    Fixture& fixture;
    Kernel::KProcess process;
    u8* backing{};
    Core::Asid asid{};
};

} // Anonymous namespace

TEST_CASE("Split tells guest memory writes from what only the GPU caches hold",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    using Writes = VideoCommon::GpuSplit::Writes;
    Fixture f;
    f.system.Initialize();
    MappedPages pages{f};
    constexpr GPUVAddr word = MappedPages::gpu_address + 0x40;
    constexpr GPUVAddr other = MappedPages::gpu_address + 0x140;
    REQUIRE_FALSE(Access::ShareWriteSlot(MappedPages::device_address + 0x40,
                                         MappedPages::device_address + 0x140));
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    split.Defer([] {}, Writes::Noted);
    split.NoteWrite(f.memory, word, 4);
    const u64 noted = Access::End(split);
    // Shader stores: a read of the caches waits for them, a read of guest memory only for the
    // writes noted there.
    split.Defer([] {}, Writes::Caches);
    CHECK(Access::WritesTarget(split, f.memory, other, 4) == Access::End(split));
    CHECK(split.MemoryWritesTarget(f.memory, word, 4) == noted);
    CHECK(split.MemoryWritesTarget(f.memory, other, 4) == 0);
    CHECK(split.LastUnplacedWrite() == 0);
    // Work nobody noted the writes of writes guest memory anywhere.
    split.Defer([] {});
    CHECK(split.LastUnplacedWrite() == Access::End(split));
    CHECK(split.MemoryWritesTarget(f.memory, other, 4) == Access::End(split));
    // A maximum's lower bound.
    split.NoteSemaphoreWrite(MappedPages::device_address + 0x40, 5, true, true);
    const auto queued = split.QueuedSemaphoreWriteAt(MappedPages::device_address + 0x40);
    REQUIRE(queued.has_value());
    CHECK(queued->known);
    CHECK(queued->at_least);
    CHECK(queued->payload == 5);
}

TEST_CASE("Split front values the releases of the QMD launches it sends as the back runs them",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    using Descriptor = Tegra::NativeQmdScheduler::Descriptor;
    using Writes = VideoCommon::GpuSplit::Writes;
    constexpr u32 Enable = 1U << 23;
    constexpr u32 OneWord = 1U << 30;
    constexpr u32 Maximum = 1U << 19;
    constexpr u32 Reduction = 1U << 26; // ADD
    constexpr GPUVAddr qmds = MappedPages::gpu_address;
    constexpr GPUVAddr words = MappedPages::gpu_address + 0x1000;
    constexpr DAddr word_devices = MappedPages::device_address + 0x1000;
    Fixture f;
    f.system.Initialize();
    Tegra::Nx2ClearGpuWrittenPages();
    MappedPages pages{f};
    Tegra::Engines::KeplerCompute compute{f.memory};
    Tegra::MacroEngine macros{true};
    std::array<u32, 128> positions{};
    compute.SetNativeMacroMode(&macros, positions);
    compute.BindRasterizer(&f.rasterizer);
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    const auto send = [&](u32 method, u32 argument) {
        compute.FrontCallMethod(f.system, split, method, argument, true, nullptr);
    };
    const auto make_qmd = [](u32 dependencies, u32 word_offset, u32 flags, u32 payload) {
        Descriptor qmd{};
        qmd[18] = 0x30;
        qmd[4] = dependencies << 14;
        qmd[24] = static_cast<u32>(words) + word_offset;
        qmd[25] = flags;
        qmd[26] = payload;
        return qmd;
    };
    // An inline QMD, in memory already (the null rasterizer's inline writes write nothing): its
    // words but the last, then, after the back ran them when `caught_up`, the last.
    const auto inline_qmd = [&](GPUVAddr address, const Descriptor& qmd, bool caught_up) {
        std::memcpy(pages.backing + (address - MappedPages::gpu_address), qmd.data(), sizeof(qmd));
        send(0xc6, static_cast<u32>((address >> 8) >> 32));
        send(0xc7, static_cast<u32>(address >> 8));
        for (u32 i = 0; i < 63; ++i) {
            send(0xc8 + i, qmd[i]);
        }
        if (caught_up) {
            Access::RunBack(split);
        }
        send(0x107, qmd[63]);
    };
    const auto word = [&](u32 offset) { return f.memory.Read<u32>(words + offset); };
    const auto queued = [&](u32 offset) {
        return split.QueuedSemaphoreWriteAt(word_devices + offset);
    };

    // A QMD that runs at once: its release is the front's once its table is the back's, which it
    // takes over when the back has run everything sent.
    inline_qmd(qmds, make_qmd(0, 0, Enable | OneWord, 5), true);
    auto release = queued(0);
    REQUIRE(release.has_value());
    CHECK(release->known);
    CHECK_FALSE(release->at_least);
    CHECK(release->payload == 5);
    Access::RunBack(split);

    // One that waits for a dependency runs at the decrement, a maximum with the word in memory.
    f.memory.Write<u32>(words + 0x10, 9);
    inline_qmd(qmds + 0x100, make_qmd(1, 0x10, Enable | OneWord | Maximum, 7), false);
    CHECK_FALSE(queued(0x10).has_value());
    send(0xad, static_cast<u32>((qmds + 0x100) >> 8));
    Access::RunBack(split);
    send(0xb0, 7);
    release = queued(0x10);
    REQUIRE(release.has_value());
    CHECK(release->known);
    CHECK_FALSE(release->at_least);
    CHECK(release->payload == 9);
    Access::RunBack(split);
    CHECK(word(0x10) == 9);

    // Over a value the front can't tell, a maximum holds its payload at least.
    split.Defer([] {}, Writes::Noted);
    split.NoteSemaphoreWrite(word_devices + 0x10, 0, false);
    inline_qmd(qmds + 0x200, make_qmd(0, 0x10, Enable | OneWord | Maximum, 12), false);
    release = queued(0x10);
    REQUIRE(release.has_value());
    CHECK(release->known);
    CHECK(release->at_least);
    CHECK(release->payload == 12);
    Access::RunBack(split);
    CHECK(word(0x10) == 12);

    // A reduction over the release before it, still queued.
    inline_qmd(qmds + 0x300, make_qmd(0, 0x20, Enable | OneWord | Maximum, 4), false);
    inline_qmd(qmds + 0x400, make_qmd(0, 0x20, Enable | OneWord | Reduction, 3), false);
    release = queued(0x20);
    REQUIRE(release.has_value());
    CHECK(release->known);
    CHECK_FALSE(release->at_least);
    CHECK(release->payload == 7);
    Access::RunBack(split);
    CHECK(word(0x20) == 7);

    // Work that may write guest memory anywhere and change the mappings, still queued: the front
    // can't tell what the back reads yet, the launch waits for it.
    split.Defer([] {});
    inline_qmd(qmds + 0x500, make_qmd(0, 0x30, Enable | OneWord, 1), false);
    CHECK_FALSE(queued(0x30).has_value());
    Access::RunBack(split);
    // The back ran everything: the front takes its table over, the waiting launch with it.
    inline_qmd(qmds + 0x600, make_qmd(0, 0x30, Enable | OneWord, 2), true);
    release = queued(0x30);
    REQUIRE(release.has_value());
    CHECK(release->payload == 2);
    Access::RunBack(split);

    // A write to a waiting QMD still queued when its decrement is sent: the launch waits, not
    // valued then.
    inline_qmd(qmds + 0x700, make_qmd(1, 0x40, Enable | OneWord | Maximum, 3), false);
    Access::RunBack(split);
    split.Defer([] {}, Writes::Noted);
    split.NoteWrite(f.memory, qmds + 0x700 + 26 * sizeof(u32), sizeof(u32));
    send(0xad, static_cast<u32>((qmds + 0x700) >> 8));
    send(0xb0, 7);
    CHECK_FALSE(queued(0x40).has_value());
    Access::RunBack(split);
    CHECK(word(0x40) == 3);
}
