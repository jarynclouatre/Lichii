// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <fmt/ranges.h>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_drain_check.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_list_indirect.h"
#include "video_core/nx2_list_indirect_stats.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_dma.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    reports_fast_draw_breaks = true;
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
    // Profiled runs take the same path: CallMethod counts and times its calls itself.
    fast_maxwell3d = !record_write_history && !trace_first_method_writes;
}

Maxwell3D::~Maxwell3D() = default;

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case RootTables::SelectMethod:
    case RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Nearly every call arrives whole in one method run (title 1 makes ~1,000,000 a second):
    // copy it to the stack instead of growing the parameter vector. Still a copy: the command
    // list is often guest memory read in place, which the guest may reuse while the macro runs
    // (and the macro memo stores the parameters after it ran). Only accurate GPU mode rewrites
    // the parameters in place (RefreshParameters), so it keeps the vector.
    // EDEN_NX2_MACRO_COPY=1: always use the vector (for comparisons).
    static const bool always_copy = std::getenv("EDEN_NX2_MACRO_COPY") != nullptr;
    if (is_last_call && macro_params.empty() && amount <= 64 && !always_copy &&
        !Settings::IsGPULevelHigh()) {
        std::array<u32, 64> parameters;
        std::copy_n(base_start, amount, parameters.begin());
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, std::span<const u32>{parameters.data(), amount});
        macro_segments.clear();
        current_macro_dirty = false;
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : gpu_addr_end - gpu_addr_begin;
        const u32 address_size = static_cast<u32>((std::min<u64>)(stream_size, (std::numeric_limits<u32>::max)()));
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::Sink};
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    const auto control = shadow_state.shadow_ram_control;
    // Sunk methods are state writes only (not in execution_mask): skip unchanged values, as
    // CallMethod does.
    const bool skip_same = SkipSameValueWrites();
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    method_sink.clear();
}

bool Maxwell3D::IsFastDrawParameter(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.size):
    case MAXWELL3D_REG_INDEX(const_buffer.address_high):
    case MAXWELL3D_REG_INDEX(const_buffer.address_low):
    case MAXWELL3D_REG_INDEX(const_buffer.offset):
    case MAXWELL3D_REG_INDEX(global_base_vertex_index):
    case MAXWELL3D_REG_INDEX(global_base_instance_index):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.format):
        return true;
    default:
        return DrawManager::IsInstanceLoopMethod(method) || IsFastDrawDeltaMethod(method) ||
               IsFastDrawVertexBuffer(method);
    }
}

bool Maxwell3D::IsFastDrawVertexBuffer(u32 method) {
    constexpr u32 streams = MAXWELL3D_REG_INDEX(vertex_streams);
    constexpr u32 stream_words = sizeof(Regs::VertexStream) / sizeof(u32);
    if (method >= streams && method < streams + Regs::NumVertexArrays * stream_words) {
        const u32 word = (method - streams) % stream_words;
        return word == 1 || word == 2; // address_high, address_low
    }
    constexpr u32 sizes = 0x600 / sizeof(u32); // SET_VERTEX_STREAM_SIZE_A/B (NativeVertexStreamSize)
    constexpr u32 limits = MAXWELL3D_REG_INDEX(vertex_stream_limits);
    constexpr u32 limit_words = sizeof(Regs::VertexStreamLimit) / sizeof(u32);
    return (method >= sizes && method < sizes + Regs::NumVertexArrays * 2) ||
           (method >= limits && method < limits + Regs::NumVertexArrays * limit_words);
}

bool Maxwell3D::IsFastDrawDeltaMethod(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + Regs::NumCBData) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    if (!DrawManager::IsInstanceLoopMethod(method) && !IsFastDrawDeltaMethod(method)) {
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Method, method);
    }
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_DEBUG(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (method) {
    case RootTables::SelectMethod:
        root_tables.selected = argument & 7;
        root_tables.offset = (argument >> 8) & 0xff;
        return;
    case RootTables::LoadMethod:
        if (root_tables.offset < 256 && root_tables.offset % 4 == 0) {
            root_tables.tables[root_tables.selected][root_tables.offset / 4] = argument;
            root_tables.loaded = true;
        }
        root_tables.offset += 4;
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    // With macros on the split GPU thread's front (FrontWrite), the front loads and binds them.
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, std::span<const u32> parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(GetNativeMacroPositions().size());

    // Execute the current macro.
    GetNativeMacroEngine().Execute(system, *this, GetNativeMacroPositions()[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::TraceFirstMethodWrite(u32 method, u32 argument) {
    static std::array<bool, Regs::NUM_REGS> seen{};
    if (method < seen.size() && !seen[method]) {
        seen[method] = true;
        LOG_INFO(HW_GPU, "3D method first write: {:#x} (byte {:#x}) = {:#x}", method, method * 4, argument);
    }
}

void Maxwell3D::DispatchMethod(Core::System& system, u32 method, u32 method_argument,
                               bool is_last_call) {
    if (!fast_maxwell3d || executing_macro != 0 || method >= MacroRegistersStart ||
        method == MAXWELL3D_REG_INDEX(shadow_ram_control)) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    if (perf_verify) {
        return VerifyCallMethod(system, method, method_argument, is_last_call);
    }
    CallMethodFast(method, method_argument, is_last_call);
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument,
                           bool is_last_call) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    // About half of the split's back's 3D writes (~5 million a second in Ocarina of Time) are state
    // registers: CallMethodFast's state write, here without the calls to it. Not in
    // execution_mask, they are not SET_SHADOW_RAM_CONTROL either.
    if (fast_maxwell3d && executing_macro == 0 && method < MacroRegistersStart &&
        !execution_mask[method] && !perf_verify &&
        !VideoCommon::Nx2Profile::Enabled()) [[likely]] {
        return WriteStateRegister(method, method_argument);
    }
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        return ProfiledCallMethod(system, method, method_argument, is_last_call);
    }
    DispatchMethod(system, method, method_argument, is_last_call);
}

// EDEN_NX2_GPU_PROFILE: the path unprofiled runs take, counted. GpuSplit's back calls the 3D engine
// outside every zone (fast_maxwell3d), millions of times a second: a sample of those calls is
// timed (SampledScope). Called inside another zone (a back macro, a run of words, the unsplit GPU
// thread's method zone), the call is that zone's time, as before.
void Maxwell3D::ProfiledCallMethod(Core::System& system, u32 method, u32 method_argument,
                                   bool is_last_call) {
    const VideoCommon::Nx2Profile::SampledScope scope{VideoCommon::Nx2Profile::Zone::Method};
    if (method < MacroRegistersStart) {
        VideoCommon::Nx2Profile::CountRegisterWrite(method,
                                                    regs.reg_array[method] == method_argument);
    }
    DispatchMethod(system, method, method_argument, is_last_call);
}

namespace {

// Executable methods the draw manager does nothing for (ProcessDirtyRegisters is all they do).
bool IsStoreOnlyExecutable(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// CallMethodSlow without the checks that cannot apply (CallMethod): no macro collects parameters,
// no write history, trace or profile, not a macro call or SET_SHADOW_RAM_CONTROL. The hot
// executable methods go straight to their handler.
void Maxwell3D::CallMethodFast(u32 method, u32 method_argument, bool is_last_call) {
    if (!execution_mask[method]) {
        return WriteStateRegister(method, method_argument);
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    ProcessDirtyRegisters(method, argument);
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method - cb_data < 16) {
        return ProcessCBData(argument);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        return draw_manager.DrawBegin(*this);
    case MAXWELL3D_REG_INDEX(draw.end):
        return draw_manager.DrawEnd(*this);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return;
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    default:
        return ProcessMethodCall(method, argument, method_argument, is_last_call);
    }
}

void Maxwell3D::VerifyCallMethod(Core::System& system, u32 method, u32 method_argument,
                                 bool is_last_call) {
    // Only register writes can run twice: handlers draw, upload and bind, and a deferred draw
    // ProcessDirtyRegisters may issue is a draw too.
    if ((execution_mask[method] && !IsStoreOnlyExecutable(method)) ||
        draw_manager.HasDeferredDraw()) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    const u32 reg = regs.reg_array[method];
    const u32 shadow = shadow_state.reg_array[method];
    const DirtyState::Flags flags = dirty.flags;
    CallMethodFast(method, method_argument, is_last_call);
    const u32 fast_reg = regs.reg_array[method];
    const u32 fast_shadow = shadow_state.reg_array[method];
    const DirtyState::Flags fast_flags = dirty.flags;
    regs.reg_array[method] = reg;
    shadow_state.reg_array[method] = shadow;
    dirty.flags = flags;
    CallMethodSlow(system, method, method_argument, is_last_call);
    if (fast_reg == regs.reg_array[method] && fast_shadow == shadow_state.reg_array[method] &&
        fast_flags == dirty.flags) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) >= 64) {
        return;
    }
    std::string flag_differences;
    const DirtyState::Flags differing = fast_flags ^ dirty.flags;
    for (size_t i = 0; i < differing.size(); ++i) {
        if (differing[i]) {
            flag_differences += fmt::format(" {}={}", i, fast_flags[i] ? 1 : 0);
        }
    }
    LOG_ERROR(HW_GPU,
              "Nx2Perf method: {:#x} = {:#x}: register {:#x} (old {:#x}), shadow {:#x} (old {:#x}), "
              "fast dirty flags differ:{}",
              method, method_argument, fast_reg, regs.reg_array[method], fast_shadow,
              shadow_state.reg_array[method], flag_differences);
}

void Maxwell3D::CallConstBufferWords(Core::System& system, const u32* methods,
                                     const u32* arguments, u32 count) {
    const u32 offset = regs.const_buffer.offset;
    const u64 bytes = u64{count} * sizeof(u32);
    const GPUVAddr address = regs.const_buffer.Address();
    // Keep per-word translation across device pages and per-word clipping at buffer limits.
    if (!fast_maxwell3d || executing_macro != 0 || address == 0 ||
        offset > regs.const_buffer.size || bytes > regs.const_buffer.size - offset ||
        bytes > Core::DEVICE_PAGESIZE - ((address + offset) & Core::DEVICE_PAGEMASK)) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        return;
    }
    std::array<u32, 64> values;
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    for (u32 i = 0; i < count; ++i) {
        values[i] = replay ? shadow_state.reg_array[methods[i]] : arguments[i];
    }
    if (perf_verify) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        std::array<u32, 64> written;
        const bool mapped = memory_manager.IsFullyMappedRange(address + offset, bytes);
        if (mapped) {
            memory_manager.ReadBlockUnsafe(address + offset, written.data(), bytes);
        }
        if (regs.const_buffer.offset != offset + bytes ||
            (mapped && std::memcmp(written.data(), values.data(), bytes) != 0)) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf method: {} constant buffer words at {:#x}+{:#x}: offset {:#x}, "
                          "expected {:#x}, memory {}",
                          count, address, offset, regs.const_buffer.offset, offset + bytes,
                          mapped && std::memcmp(written.data(), values.data(), bytes) != 0
                              ? "differs"
                              : "matches");
            }
        }
        return;
    }
    // Counted and timed as CallMethod counts and times each word (EDEN_NX2_GPU_PROFILE).
    const VideoCommon::Nx2Profile::SampledScope scope{VideoCommon::Nx2Profile::Zone::Method, count};
    const bool profile = VideoCommon::Nx2Profile::Enabled();
    // As CallMethodFast for each word: the first may issue a deferred draw, before any upload.
    for (u32 i = 0; i < count; ++i) {
        if (profile) [[unlikely]] {
            VideoCommon::Nx2Profile::CountRegisterWrite(methods[i],
                                                        regs.reg_array[methods[i]] == arguments[i]);
        }
        ProcessDirtyRegisters(methods[i], ProcessShadowRam(methods[i], arguments[i]));
    }
    const VideoCommon::Nx2Profile::FineScope upload_scope{VideoCommon::Nx2Profile::Zone::CbUpload};
    if (VideoCommon::Nx2FastDraw::Enabled()) {
        NoteFastDrawConstantData(address + offset, address + offset + bytes);
    }
    memory_manager.WriteWordsCached(address + offset, values.data(), count);
    regs.const_buffer.offset += static_cast<u32>(bytes);
}

void Maxwell3D::CallMethodSlow(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // Every word of a macro register run is a parameter of the macro it started, whichever macro
    // register it writes (as on the split front, FrontCallMethod). Only a write to an ordinary
    // register before the macro has all its parameters is an error.
    if (executing_macro != 0 && method < MacroRegistersStart) {
        DumpRecentWrites(method);
    }
    if (executing_macro != 0) {
        ASSERT(method >= MacroRegistersStart);
    }
    RecordWrite(method, method_argument, 1, is_last_call ? 1 : 2);

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }
    if (trace_first_method_writes) [[unlikely]] {
        TraceFirstMethodWrite(method, method_argument);
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && front_macros) [[unlikely]] {
        back_control_writes[back_control_count++ % back_control_writes.size()] = {
            method_argument, static_cast<u32>(shadow_state.shadow_ram_control), is_last_call ? 1U : 0U};
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    // A state register written with the value it holds changes nothing: no dirty flags (they
    // would make the next draw re-check that state), no method processing. Title 1 makes ~6
    // million 3D register writes a second, ~60% of them same-value, most from macro replays.
    // Methods with side effects (execution_mask: draws, clears, constant buffer data and
    // binds, semaphores, ...) always run. EDEN_NX2_SKIP_SAME=0 turns it off.
    if (!execution_mask[method]) {
        // ProcessMethodCall (and the draw manager behind it) acts only on executable methods.
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::DumpRecentWrites(u32 method) {
    static u32 dumps{};
    if (dumps++ >= 4) {
        return;
    }
    LOG_ERROR(HW_GPU, "Write to {:#x} while macro {:#x} collects parameters ({} so far); last writes:",
              method, executing_macro, macro_params.size());
    if (!record_write_history) {
        LOG_ERROR(HW_GPU, "Register write history disabled; EDEN_NX2_3D_HISTORY=1 enables it");
        return;
    }
    for (size_t i = 0; i < recent_writes.size(); ++i) {
        const auto& w = recent_writes[(recent_write_index + i) % recent_writes.size()];
        if (w.amount != 0) {
            LOG_ERROR(HW_GPU, "  method {:#x} arg {:#x} amount {} pending {}", w.method, w.argument,
                      w.amount, w.pending);
        }
    }
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    RecordWrite(method, amount ? base_start[0] : 0, amount, methods_pending);
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // As in ProcessDirtyRegisters: the paths below write without it.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::InlineData);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    const auto positions = GetNativeMacroPositions();
    const u32 index = regs.load_mme.start_address_ptr++;
    if (index < positions.size()) {
        positions[index] = data;
    }
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
    if (const u64 trace = VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::Write)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY write n={} path=stamp as={} gpu={:#x} bytes={} value={:#x}",
                 trace, memory_manager.GetID(), sequence_address, long_query ? 8 : 4,
                 long_query ? payload : static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::QueryReport};
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::RenderEnable};
    if (rasterizer->AccelerateConditionalRendering()) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    Regs::ReportSemaphore::Compare cmp{};
    bool operands_read = false;
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
    if (const u64 trace =
            VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::CpuCondition)) {
        LOG_INFO(HW_GPU,
                 "DKB_CONDITION cpu n={} as={} gpu={:#x} mode={} override={} execute={} "
                 "operands_read={} initial={:#x}/{:#x} current={:#x}/{:#x}",
                 trace, memory_manager.GetID(), condition_address,
                 static_cast<u32>(regs.render_enable.mode),
                 static_cast<u32>(regs.render_enable_override), execute_on, operands_read,
                 cmp.initial_sequence, cmp.initial_mode, cmp.current_sequence, cmp.current_mode);
    }
}

void Maxwell3D::ProcessCounterReset() {
    const VideoCommon::Nx2Profile::FineScope profile_scope{
        VideoCommon::Nx2Profile::Zone::CounterReset};
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    if (GetNativeMacroEngine().native_turing) {
        LOG_DEBUG(HW_GPU, "Native 3D syncpoint increment: id={}, raw={:#x}", regs.sync_info.sync_point.Value(), regs.reg_array[MAXWELL3D_REG_INDEX(sync_info)]);
    }
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::CbBind};
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    static std::array<std::array<bool, 32>, 5> seen{};
    if (!seen[stage_index][bind_data.shader_slot]) {
        seen[stage_index][bind_data.shader_slot] = true;
        LOG_INFO(HW_GPU, "Constant buffer bind: stage {} slot {} address {:#x} size {:#x}", stage_index,
                 bind_data.shader_slot.Value(), regs.const_buffer.Address(), regs.const_buffer.size);
    }
    if (bind_data.shader_slot >= Regs::MaxConstBuffers) {
        return; // slot field is 5 bits; the state array holds 18
    }
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    if (VideoCommon::Nx2FastDraw::Enabled() &&
        (buffer.enabled != (bind_data.valid.Value() != 0) ||
         buffer.address != regs.const_buffer.Address() || buffer.size != regs.const_buffer.size)) {
        VideoCommon::Nx2FastDraw::NoteBind(stage_index, bind_data.shader_slot);
    }
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::NoteFastDrawConstantData(GPUVAddr begin, GPUVAddr end) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    // Texture and sampler headers are read when image views are resolved, which delta draws skip.
    constexpr u64 HeaderSize = 0x20;
    const GPUVAddr tic = regs.tex_header.Address();
    const u64 tic_size = (u64{regs.tex_header.limit} + 1) * HeaderSize;
    // As TextureCache::SynchronizeDescriptors sizes the sampler table.
    const GPUVAddr tsc = regs.tex_sampler.Address();
    const u64 tsc_size = (u64{regs.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding
                                  ? regs.tex_header.limit
                                  : regs.tex_sampler.limit} +
                          1) *
                         HeaderSize;
    if ((begin < tic + tic_size && tic < end) || (begin < tsc + tsc_size && tsc < end)) {
        FastDraw::Break(FastDraw::Reason::TextureHeaders);
        return;
    }
    FastDraw::NoteConstantData(begin, end);
    // The same device memory can be mapped at other GPU addresses: compare where it lands.
    fast_draw_header_spans.Update(memory_manager, tic, tic_size);
    fast_draw_sampler_spans.Update(memory_manager, tsc, tsc_size);
    constexpr u64 PageSize = 0x1000; // the smallest GPU page
    for (GPUVAddr page = begin; page < end;) {
        const GPUVAddr page_end = (std::min)(end, (page & ~(PageSize - 1)) + PageSize);
        if (const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(page)) {
            const DAddr device_end = *device + (page_end - page);
            if (fast_draw_header_spans.Intersects(*device, device_end) ||
                fast_draw_sampler_spans.Intersects(*device, device_end)) {
                FastDraw::Break(FastDraw::Reason::TextureHeaders);
                return;
            }
            FastDraw::NoteDeviceData(*device, device_end);
        }
        page = page_end;
    }
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::CbUpload};
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    const size_t copy_size = amount * sizeof(u32);
    const size_t room = regs.const_buffer.offset < regs.const_buffer.size
                            ? regs.const_buffer.size - regs.const_buffer.offset
                            : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        if (VideoCommon::Nx2FastDraw::Enabled()) {
            NoteFastDrawConstantData(address, address + write_size);
        }
        memory_manager.WriteBlockCached(address, start_base, write_size);
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      regs.const_buffer.offset, regs.const_buffer.size, copy_size, write_size);
        }
    }

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2). The front keeps
// what macros read: the registers as the back thread will hold them when it reaches the same
// point of the method stream. That is every write the front sends, with the shadow RAM applied,
// plus the few changes the back makes on its own (constant buffer offset, shadow_scratch[0] on
// falcon[4], the macro bind pointer); the front loads and binds the macro code itself. Macros
// that read memory run on the back instead (FrontRunOnBack), and the registers they can write
// are the back's until the front writes them or reads them back (FrontRegister).
struct Maxwell3D::FrontState {
    std::array<ControlWrite, 16> control_writes{};
    u32 control_count{};
    std::array<u32, Regs::NUM_REGS> regs{};
    std::array<u32, Regs::NUM_REGS> shadow{};
    Regs::ShadowRamControl control{};
    u32 executing_macro{};
    std::vector<u32> parameters;
    u64 calls{};
    /// A method the draw manager handles was sent since the back last ran DrawDeferred: only
    /// then can it have a deferred instanced draw to issue at the end of a macro.
    bool draw_pending{true};
    /// The back holds state writes in its method sink, which its next executable method consumes
    /// (issuing a deferred draw a changed one ends). Front() consumes it before copying the state.
    bool sink_pending{};
    u32 split_start{}; ///< GpuSplit::Starts() when this copy was made
    bool skip_same{}; ///< SkipSameValueWrites() && FrontSkipSame(), read once
    /// The back's draw manager is in an instance loop (DrawMode::Instance): draw.end does nothing.
    bool instance_loop{};

    /// Registers a macro the back ran may have written (FrontRunOnBack), and their shadow RAM
    /// entries: the ring position after the last such macro, 0 once the front knows the value.
    std::array<u64, Regs::NUM_REGS> back_regs{};
    std::array<u64, Regs::NUM_REGS> back_shadow{};
    std::bitset<Regs::NUM_REGS> back_owned; ///< back_regs or back_shadow set
    struct BackMacro {
        u32 start{};
        bool back{};                           ///< Runs on the back (else on the front)
        std::vector<u32> writes;               ///< The registers its code can write
        std::bitset<Regs::NUM_REGS> write_set; ///< The same, for the back's check
        std::atomic<bool> escaped{};           ///< The back saw it write another register
        u32 checked{};                         ///< Calls EDEN_NX2_BACK_MACRO_VERIFY checked
    };
    std::vector<std::pair<u32, u32>> back_record; ///< A checked call's writes on the back
    /// By start in the macro code of back_generation, once called.
    std::vector<std::unique_ptr<BackMacro>> back_macros;
    u64 back_generation{~u64{0}};
    std::vector<u32> back_demoted; ///< Starts of macros that wrote outside their write set
    /// The back's macro engine, its code a copy of the front's (when back_code_copied).
    std::unique_ptr<MacroEngine> back_engine;
    bool back_code_copied{};
    u64 back_last{};                 ///< The position after the last macro sent to the back
    std::atomic<u32> back_escapes{}; ///< Back macros that wrote outside their write set
    u32 back_escapes_seen{};
};

namespace {

// EDEN_NX2_SPLIT_VERIFY=1: every 4096 front macro calls, wait for the back and compare the
// front's registers with the back's (logs and resynchronizes differences).
bool FrontVerifyEnabled() {
    static const bool verify = std::getenv("EDEN_NX2_SPLIT_VERIFY") != nullptr;
    return verify;
}

/// A macro DMA read of `words` at `source`: whether the caches held them (`drained`), and else
/// whether it waited for queued writers and what it read before the drain (`undrained`).
void CheckDmaRead(u32 macro, GPUVAddr source, bool drained, bool waited,
                  std::span<const u32> undrained, std::span<const u32> words) {
    static u64 reads{};
    static u64 undrained_reads{};
    static u64 waits{};
    static u64 differences{};
    static u64 total_differences{};
    static std::chrono::steady_clock::time_point last_report{};
    ++reads;
    if (!drained) {
        ++undrained_reads;
        waits += waited ? 1 : 0;
        if (!std::ranges::equal(undrained, words)) {
            ++differences;
            if (++total_differences <= 16) {
                const size_t shown = std::min<size_t>(words.size(), 8);
                LOG_WARNING(HW_GPU,
                            "Macro DMA read check: macro {:#x} read {} words at {:#x} as {:08x} "
                            "without draining the split, as {:08x} after it",
                            macro, words.size(), source, fmt::join(undrained.first(shown), ","),
                            fmt::join(words.first(shown), ","));
            }
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (last_report == std::chrono::steady_clock::time_point{}) {
        last_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    if (seconds < 10.0) {
        return;
    }
    last_report = now;
    LOG_INFO(HW_GPU,
             "Macro DMA read checks in {:.1f} s: {} reads, {} without draining the split ({} "
             "waited for queued writers), {} of those with other words than after a drain ({} "
             "in all)",
             seconds, reads, undrained_reads, waits, differences, total_differences);
    reads = 0;
    undrained_reads = 0;
    waits = 0;
    differences = 0;
}

// Executable methods ProcessMethodCall handles itself; the rest go to the draw manager.
bool FrontHandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

// EDEN_NX2_SPLIT_SKIP=0: send state writes that keep the register's value too.
bool FrontSkipSame() {
    static const bool skip = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    return skip;
}

// EDEN_NX2_BACK_MACROS=0: macros that read memory run on the front too (FrontRunOnBack).
bool BackMacrosEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_BACK_MACROS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// Whether a macro the split's back runs may write `method` (FrontRunOnBack): the front then only
// loses track of the register's value. Not the methods with effects the front tracks (macro code
// and bindings, the shadow RAM mode, draws, constant buffer data, reports, uploads, syncpoints),
// nor the registers it reads to note their memory writes, nor registers the back changes itself,
// nor MME_DMA_READ, whose words land in the front's data RAM.
bool BackWritable(const Maxwell3D& maxwell3d, u32 method) {
    constexpr u32 const_buffer = MAXWELL3D_REG_INDEX(const_buffer);
    constexpr u32 report = MAXWELL3D_REG_INDEX(report_semaphore);
    constexpr u32 upload = MAXWELL3D_REG_INDEX(upload);
    constexpr u32 upload_end = MAXWELL3D_REG_INDEX(launch_dma);
    if (method - const_buffer < 4 || method - report < 4 || method - upload < upload_end - upload) {
        return false;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(load_mme.start_address_ptr): // Advanced by load_mme.start_address
    case MAXWELL3D_REG_INDEX(shadow_scratch):             // Set by falcon[4]
    case MAXWELL3D_REG_INDEX(logic_op.enable):            // Cleared by draws on AMD
    case 0x157:
        return false;
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true; // Executable, with effects on the back only
    default:
        return method < MacroRegistersStart && !maxwell3d.execution_mask[method];
    }
}

// EDEN_NX2_BACK_MACRO_VERIFY=<n> (64 without a number): the first n calls of each macro the back
// runs (FrontRunOnBack) also run the old way, on the front once the split is drained
// (FrontRunTheOldWay), and the front compares the writes (method, value) both make; they must
// not differ. Every 10 s it logs the counts, with the front's reads and replays of registers
// such macros wrote and how many of them waited for the back.
u32 BackMacroChecks() {
    static const u32 checks = [] {
        const char* value = std::getenv("EDEN_NX2_BACK_MACRO_VERIFY");
        if (value == nullptr) {
            return 0U;
        }
        const unsigned long calls = std::strtoul(value, nullptr, 10);
        return calls != 0 ? static_cast<u32>(calls) : 64U;
    }();
    return checks;
}

struct BackMacroStats {
    u64 calls{};
    u64 checked{};
    u64 differences{};
    u64 total_differences{};
    u64 reads{};
    u64 read_waits{};
    u64 replays{};
    u64 replay_waits{};
    std::chrono::steady_clock::time_point last_report{};
};

BackMacroStats& BackStats() {
    static BackMacroStats stats;
    return stats;
}

void ReportBackMacroStats() {
    auto& stats = BackStats();
    const auto now = std::chrono::steady_clock::now();
    if (stats.last_report == std::chrono::steady_clock::time_point{}) {
        stats.last_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - stats.last_report).count();
    if (seconds < 10.0) {
        return;
    }
    LOG_INFO(HW_GPU,
             "Split back macros in {:.1f} s: {} calls, {} checked against the old way, {} of "
             "those with other writes ({} in all); {} front reads of registers they wrote ({} "
             "waited), {} replays of their shadow RAM entries ({} waited)",
             seconds, stats.calls, stats.checked, stats.differences, stats.total_differences,
             stats.reads, stats.read_waits, stats.replays, stats.replay_waits);
    stats = BackMacroStats{.total_differences = stats.total_differences, .last_report = now};
}

std::string DescribeWrites(std::span<const std::pair<u32, u32>> writes) {
    std::string text;
    for (size_t i = 0; i < std::min<size_t>(writes.size(), 12); ++i) {
        text += fmt::format("{}{:#x}={:#x}", i == 0 ? "" : ",", writes[i].first, writes[i].second);
    }
    return writes.size() > 12 ? text + ",..." : text;
}

// GPU-written list words the channel's compute macros left in its data RAM (Nx2ListIndirect):
// the front runs 3D macros on the words themselves, so a macro at `start` that may read one
// gets them first, read back from their slots.
void ConcretizeListWords(VideoCommon::GpuSplit& split, VideoCore::RasterizerInterface* rasterizer,
                         Nx2ListIndirect::DataRamWords& channel, std::span<u32> data_ram,
                         const MacroEngine& engine, u32 start) {
    const auto& reads =
        channel.reads_3d.Get(engine.native_code, engine.native_code_generation, start);
    if (!Nx2ListIndirect::MayRead(reads.data_ram_known, reads.data_ram, channel.words)) {
        return;
    }
    const auto release = [&](u32 slot) { Nx2ListIndirect::ReleaseSlot(split, rasterizer, slot); };
    Nx2ListIndirect::Prune(channel.words, [data_ram](u32 word) { return data_ram[word]; }, release);
    if (!Nx2ListIndirect::MayRead(reads.data_ram_known, reads.data_ram, channel.words)) {
        return;
    }
    Nx2ListIndirect::Stats::NoteConcretized(Nx2ListIndirect::Reason::Macro3d); // TEST
    Nx2ListIndirect::ConcretizeDataRam(
        channel.words, data_ram,
        [&](std::span<const u32> slots, std::span<u32> words) {
            const u64 read_start = Nx2ListIndirect::Stats::NowNs(); // TEST
            Nx2ListIndirect::ReadSlots(split, rasterizer, slots, words);
            Nx2ListIndirect::Stats::NoteRead(slots.size(),
                                             Nx2ListIndirect::Stats::NowNs() - read_start);
        },
        release);
}

} // Anonymous namespace

Maxwell3D::FrontState& Maxwell3D::Front(Core::System& system, VideoCommon::GpuSplit& split) {
    // Copied during an earlier start: while the split was stopped, state writes went to the
    // method sink without reaching it (they hand the front back only once consumed).
    if (front && front->split_start != split.Starts()) [[unlikely]] {
        split.Drain(VideoCommon::DrainSite::FrontStart); // The engine's state is the back's.
        ReleaseFront();
    }
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle (state writes it
        // still holds in its method sink first).
        split.Drain(VideoCommon::DrainSite::FrontStart);
        ConsumeSink(system);
        front = std::make_unique<FrontState>();
        std::memcpy(front->regs.data(), regs.reg_array.data(), sizeof(front->regs));
        std::memcpy(front->shadow.data(), shadow_state.reg_array.data(), sizeof(front->shadow));
        front->control = shadow_state.shadow_ram_control;
        front->skip_same = SkipSameValueWrites() && FrontSkipSame();
        front->split_start = split.Starts();
        front->executing_macro = executing_macro;
        front->parameters = std::move(macro_params);
        macro_params.clear();
        executing_macro = 0;
        front_macros = true;
    }
    return *front;
}

// With the split on, engine methods run on its back only: one here on another thread means the
// split stopped (accuracy settings changed mid-game). Hand the macros back: the front's copy
// goes stale from now on, and a restart copies the registers again (Front).
void Maxwell3D::ReleaseFront() {
    // The macros run on the engine from now on: the data RAM gets the GPU-written list words
    // its words hold placeholders of (Nx2ListIndirect).
    if (auto* channel = Nx2ListIndirect::FindChannel(&native_data_ram);
        channel != nullptr && !channel->words.empty()) {
        auto& split = VideoCommon::GpuSplit::Get();
        Nx2ListIndirect::Stats::NoteConcretized(Nx2ListIndirect::Reason::Stop); // TEST
        Nx2ListIndirect::ConcretizeDataRam(
            channel->words, std::span<u32>{native_data_ram},
            [&](std::span<const u32> slots, std::span<u32> words) {
                Nx2ListIndirect::ReadSlots(split, rasterizer, slots, words);
            },
            [&](u32 slot) { Nx2ListIndirect::ReleaseSlot(split, rasterizer, slot); });
    }
    executing_macro = front->executing_macro;
    macro_params = std::move(front->parameters);
    // Entries macros the back ran may have written are the engine's already.
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (front->back_shadow[i] == 0) {
            shadow_state.reg_array[i] = front->shadow[i];
        }
    }
    shadow_state.shadow_ram_control = front->control;
    front.reset();
    front_macros = false;
    LOG_INFO(HW_GPU, "GPU thread split stopped: 3D macros back on the engine");
}

void Maxwell3D::FrontWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                           u32 argument, bool is_last_call, bool from_macro) {
    namespace Profile = VideoCommon::Nx2Profile;
    // As ProcessShadowRam does on the back. A register a macro the back ran may have written is
    // the front's again, and its write is sent even if the front's copies hold the value.
    u32 value = argument;
    bool shadow_kept = !(fs.back_owned[method] && FrontTakeOver(split, fs, method));
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_kept = shadow_kept && fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
        fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
            argument, static_cast<u32>(fs.control), from_macro ? 2U : 1U};
    }
    if (execution_mask[method] && FrontSkipsLoopWrite(fs, method, value, shadow_kept)) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        return;
    }
    if (!execution_mask[method]) {
        // Nothing would change on the back: CallMethod and ConsumeSinkImpl skip same-value state
        // writes, and the shadow RAM (tracked by default) holds the value already.
        if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            return;
        }
        fs.regs[method] = value;
    } else {
        fs.regs[method] = value;
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        switch (method) {
        case MAXWELL3D_REG_INDEX(shadow_ram_control):
            // The back's mode is shadow_state.shadow_ram_control: the shadow copy of this very
            // register, which a later Replay write reads back.
            fs.control = static_cast<Regs::ShadowRamControl>(argument);
            fs.shadow[method] = argument;
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
            GetNativeMacroEngine().ClearCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)]);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction):
            GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.start_address): {
            const auto positions = GetNativeMacroPositions();
            u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
            if (pointer < positions.size()) {
                positions[pointer] = value;
            }
            ++pointer;
            break;
        }
        case MAXWELL3D_REG_INDEX(falcon[4]):
            fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
            break;
        default:
            if (method >= cb_data && method < cb_data + 16) {
                fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += sizeof(u32);
            } else if (!FrontHandledOutsideDrawManager(method)) {
                fs.draw_pending = true;
            }
            break;
        }
    }
    using Writes = VideoCommon::GpuSplit::Writes;
    if (from_macro) {
        // Direct writes consume the sink first.
        split.MacroWrite(this, method, argument, Writes::Noted);
        fs.sink_pending = false;
    } else {
        split.Method(this, method, argument, is_last_call, nullptr, Writes::Noted);
        fs.sink_pending = !execution_mask[method];
    }
    // A write the back only keeps for later (its method sink) runs nothing there yet.
    if (from_macro || execution_mask[method]) {
        FrontNoteWrites(split, fs, method, 1);
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        FrontNoteReport(split, fs);
    }
}

void Maxwell3D::FrontNoteReport(VideoCommon::GpuSplit& split, const FrontState& fs) {
    // What ProcessQueryGet will write on the back: a payload report writes the payload, a counter
    // what the GPU counted.
    Regs::ReportSemaphore report;
    std::memcpy(&report, &fs.regs[MAXWELL3D_REG_INDEX(report_semaphore)], sizeof(report));
    const auto operation = report.query.operation.Value();
    if (operation != Regs::ReportSemaphore::Operation::Release &&
        operation != Regs::ReportSemaphore::Operation::ReportOnly) {
        return;
    }
    const auto device = memory_manager.GpuToCpuAddress(report.Address());
    if (!device) {
        return;
    }
    const bool payload = report.query.report == Regs::ReportSemaphore::Report::Payload;
    // As QueryCacheBase::CounterReport writes it: a payload report without a fence flag as the
    // back runs it, a short release (fence-flagged) or a counter once a fence ran.
    const bool fence_flagged = operation == Regs::ReportSemaphore::Operation::Release &&
                               report.query.short_query != 0;
    const auto landing = payload && !fence_flagged ? VideoCommon::GpuSplit::Landing::AtRun
                                                   : VideoCommon::GpuSplit::Landing::Fenced;
    split.NoteSemaphoreWrite(*device, report.payload, payload, false, landing);
    VideoCommon::Nx2AcquireStats::NoteRelease(*device,
                                              VideoCommon::Nx2AcquireStats::Releaser::Report3D);
    if (report.query.short_query == 0) {
        // 64 bits (a timestamp follows).
        split.NoteSemaphoreWrite(*device + 4, 0, payload, false, landing);
    }
}

void Maxwell3D::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                u32 argument, bool is_last_call) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontWrite(split, fs, method, argument, is_last_call, false);
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    // What CallMultiMethod does to the registers on the back, then one entry for all the words
    // (sent whatever the front's copies hold).
    if (fs.back_owned[method]) {
        FrontTakeOver(split, fs, method);
    }
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    if (method >= cb_data && method < cb_data + 16) {
        fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += amount * sizeof(u32);
    } else if (method == MAXWELL3D_REG_INDEX(inline_data)) {
        // Upload data only.
    } else if ((method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        FrontTrackInstanceLoop(fs, method, 0);
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
                fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
                    argument, static_cast<u32>(fs.control), 3U};
            }
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            FrontTrackInstanceLoop(fs, method, value); // DrawBegin runs for every word
            if (!execution_mask[method]) {
                continue;
            }
            switch (method) {
            case MAXWELL3D_REG_INDEX(shadow_ram_control):
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
                GetNativeMacroEngine().ClearCode(value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction):
                GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.start_address): {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case MAXWELL3D_REG_INDEX(falcon[4]):
                fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
                break;
            default:
                if (!FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                break;
            }
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr,
                      VideoCommon::GpuSplit::Writes::Noted);
    fs.sink_pending = false; // The back consumes the sink before a run.
    FrontNoteWrites(split, fs, method, amount);
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount != 0) {
        FrontNoteReport(split, fs);
    }
}

void Maxwell3D::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                              FrontState& fs) {
    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    VideoCommon::Nx2Profile::CountMethod(method, false);
    if (!FrontRunOnBack(system, split, fs, engine, start)) {
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
        if (auto* channel = Nx2ListIndirect::FindChannel(&native_data_ram);
            channel != nullptr && !channel->words.empty()) {
            ConcretizeListWords(split, rasterizer, *channel, native_data_ram, engine, start);
        }
        engine.native_interpreter.active_dram = &native_data_ram;
        const auto parameter_count = fs.parameters.size();
        const auto read = [&](u32 address) {
            if (address >= Regs::NUM_REGS) {
                return 0U;
            }
            return fs.back_owned[address] ? FrontRegister(split, fs, address) : fs.regs[address];
        };
        const auto emit = [&](u32 address, u32 value) {
            if (address >= MacroRegistersStart) {
                return false;
            }
            FrontWrite(split, fs, address, value, true, true);
            // MME_DMA_READ (into data RAM) and MME_DMA_READ_FIFOED (into the load FIFO).
            if (address == 0x157 || address == 0x158) {
                FrontDmaRead(split, fs, method, address, value);
            }
            return true;
        };
#ifdef ARCHITECTURE_x86_64
        if (!engine.native_jit) {
            engine.native_jit = std::make_shared<TuringMacroJit>();
        }
        engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
        const auto result = engine.native_jit->Execute(
            start, fs.parameters, std::span<u32>{native_data_ram}, read, emit);
#else
        const auto result =
            engine.native_interpreter.Execute(engine.native_code, start, fs.parameters, read, emit);
#endif
        if (!result.success) {
            engine.native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Native MME (split front) failed: start={:#x}, consumed={}/{}, error={}",
                      start, result.consumed, parameter_count, result.error);
        }
    }
    fs.parameters.clear();
    if (fs.draw_pending) {
        fs.draw_pending = false;
        // A deferred draw.
        split.Defer([this, system_ptr = &system] { FinishFrontMacro(*system_ptr); },
                    VideoCommon::GpuSplit::Writes::Caches, VideoCommon::GpuSplit::FastDrawNeutral);
    }
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

// A macro whose code reads memory into its FIFO (MME_DMA_READ_FIFOED) and writes only registers
// BackWritable allows runs on the back, at its point of the method stream: it reads the words as
// a drained front would, and the front goes on without waiting for the work queued before it.
// The registers it can write (and, in a tracked shadow RAM mode, their entries) are the back's
// until the front writes them (FrontTakeOver) or needs them (FrontRegister). Returns whether the
// call went to the back.
bool Maxwell3D::FrontRunOnBack(Core::System& system, VideoCommon::GpuSplit& split, FrontState& fs,
                               const MacroEngine& engine, u32 start) {
    using DrainSite = VideoCommon::DrainSite;
    using BackMacro = FrontState::BackMacro;
    if (!BackMacrosEnabled() || fs.parameters.size() > 0x1000) {
        return false;
    }
    // A macro the back ran wrote another register: the front's copies may be stale anywhere. It
    // runs on the front from now on, and the front takes the back's registers (FrontVerify).
    const auto take_escapes = [&] {
        fs.back_escapes_seen = fs.back_escapes.load(std::memory_order_acquire);
        for (const auto& macro : fs.back_macros) {
            if (macro && macro->back && macro->escaped.load(std::memory_order_relaxed)) {
                macro->back = false;
                fs.back_demoted.push_back(macro->start);
                LOG_ERROR(HW_GPU,
                          "Split back macro at {:#x} wrote outside the registers its code "
                          "can write; it runs on the front from now on",
                          macro->start);
            }
        }
        FrontVerify(split, fs);
    };
    if (fs.back_generation != engine.native_code_generation) [[unlikely]] {
        // The macros sent before run the code they were sent with, and check their records.
        split.WaitForPosition(fs.back_last, DrainSite::BackCode);
        if (fs.back_escapes.load(std::memory_order_relaxed) != fs.back_escapes_seen) {
            take_escapes();
        }
        fs.back_macros.clear();
        fs.back_macros.resize(engine.native_code.size() / 3);
        fs.back_generation = engine.native_code_generation;
        fs.back_code_copied = false;
    } else if (fs.back_escapes.load(std::memory_order_relaxed) != fs.back_escapes_seen)
        [[unlikely]] {
        take_escapes();
    }
    if (start >= fs.back_macros.size()) {
        return false;
    }
    auto& known = fs.back_macros[start];
    if (!known) [[unlikely]] {
        known = std::make_unique<BackMacro>();
        known->start = start;
        const auto effects = TuringMacroInterpreter::AnalyzeEffects(engine.native_code, start);
        const auto& methods = effects.methods;
        const bool fifo_read = std::ranges::binary_search(methods, 0x158U);
        if (fifo_read || std::ranges::binary_search(methods, 0x157U)) {
            std::string reason;
            if (std::ranges::find(fs.back_demoted, start) != fs.back_demoted.end()) {
                reason = "it wrote another register on the back before";
            } else if (!effects.methods_known) {
                reason = "it computes the method of a write";
            } else if (effects.data_ram) {
                reason = "it uses the data RAM";
            } else if (const auto it = std::ranges::find_if(
                           methods, [this](u32 m) { return !BackWritable(*this, m); });
                       it != methods.end()) {
                reason = fmt::format("it writes {:#x}", *it);
            }
            known->back = reason.empty() && fifo_read;
            if (known->back) {
                known->writes = methods;
                for (const u32 written : methods) {
                    known->write_set.set(written);
                }
            }
            static u32 logged{};
            if (logged < 32) {
                ++logged;
                LOG_INFO(HW_GPU, "Split macro at {:#x} reads memory and runs on the {}: {}", start,
                         known->back ? "back" : "front",
                         known->back ? fmt::format("writes {:#x}", fmt::join(methods, ","))
                                     : reason);
            }
        }
    }
    if (!known->back) {
        return false;
    }
    if (!fs.back_engine) {
        fs.back_engine = std::make_unique<MacroEngine>(true);
        fs.back_engine->native_turing = true;
    }
    MacroEngine& back = *fs.back_engine;
    if (!fs.back_code_copied) {
        // Nothing sent with the old code is left to run (waited for above).
        back.native_code = engine.native_code;
        back.native_code_generation = engine.native_code_generation;
        fs.back_code_copied = true;
    }
    const bool check = known->checked < BackMacroChecks();
    std::vector<std::pair<u32, u32>> old_writes;
    if (check) [[unlikely]] {
        ++known->checked;
        old_writes = FrontRunTheOldWay(split, fs, start);
        fs.back_record.clear();
        back.split_record = &fs.back_record;
    }
    split.DeferWords(
        [this, system_ptr = &system, engine_ptr = &back, macro = known.get(),
         escapes = &fs.back_escapes](std::span<const u32> parameters) {
            // As ProcessMacro calls a macro, with the back's engine.
            ConsumeSink(*system_ptr);
            engine_ptr->split_writes = &macro->write_set;
            const u64 unexpected = engine_ptr->split_unexpected_writes;
            engine_ptr->Execute(*system_ptr, *this, macro->start, parameters);
            if (engine_ptr->split_unexpected_writes != unexpected) [[unlikely]] {
                macro->escaped.store(true, std::memory_order_relaxed);
                escapes->fetch_add(1, std::memory_order_release);
            }
        },
        // Its writes (BackWritable) touch registers and the caches, not guest memory.
        fs.parameters, VideoCommon::GpuSplit::Writes::Caches);
    fs.sink_pending = false; // The back macro consumes the sink before it runs.
    const u64 position = split.Position();
    fs.back_last = position;
    const bool tracked = fs.control == Regs::ShadowRamControl::Track ||
                         fs.control == Regs::ShadowRamControl::TrackWithFilter;
    for (const u32 written : known->writes) {
        fs.back_regs[written] = position;
        if (tracked) {
            fs.back_shadow[written] = position;
        }
        fs.back_owned[written] = true;
    }
    if (BackMacroChecks() != 0) [[unlikely]] {
        auto& stats = BackStats();
        ++stats.calls;
        if (check) {
            split.Drain(DrainSite::FrontVerify);
            back.split_record = nullptr;
            ++stats.checked;
            if (old_writes != fs.back_record) {
                ++stats.differences;
                if (++stats.total_differences <= 16) {
                    LOG_WARNING(HW_GPU,
                                "Split back macro check: macro at {:#x} wrote {} the old way, {} "
                                "on the back",
                                start, DescribeWrites(old_writes), DescribeWrites(fs.back_record));
                }
            }
        }
        ReportBackMacroStats();
    }
    return true;
}

// EDEN_NX2_BACK_MACRO_VERIFY: the writes (method, value) the call FrontRunOnBack sends makes run
// the old way: on the front, against its copies of the registers, with the split drained first
// and its DMA reads made once the caches wrote the words out. Registers macros the back ran may
// have written are the back's, exact once drained; the call's own writes are what the back's
// CallMethod leaves (the shadow RAM applied). Writes nothing itself.
std::vector<std::pair<u32, u32>> Maxwell3D::FrontRunTheOldWay(VideoCommon::GpuSplit& split,
                                                              const FrontState& fs, u32 start) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    std::vector<std::pair<u32, u32>> writes;
    std::unordered_map<u32, u32> values;
    std::unordered_map<u32, u32> entries;
    const auto value_of = [&](u32 method) {
        const auto it = values.find(method);
        return it != values.end()          ? it->second
               : fs.back_regs[method] != 0 ? regs.reg_array[method]
                                           : fs.regs[method];
    };
    const auto entry_of = [&](u32 method) {
        const auto it = entries.find(method);
        return it != entries.end()           ? it->second
               : fs.back_shadow[method] != 0 ? shadow_state.reg_array[method]
                                             : fs.shadow[method];
    };
    std::vector<u32> fifo = fs.parameters;
    const auto read = [&](u32 address) {
        return address < Regs::NUM_REGS ? value_of(address) : 0U;
    };
    const auto emit = [&](u32 address, u32 value) {
        if (address >= MacroRegistersStart) {
            return false;
        }
        writes.emplace_back(address, value);
        u32 stored = value;
        if (fs.control == Regs::ShadowRamControl::Track ||
            fs.control == Regs::ShadowRamControl::TrackWithFilter) {
            entries[address] = value;
        } else if (fs.control == Regs::ShadowRamControl::Replay) {
            stored = entry_of(address);
        }
        values[address] = stored;
        if (address == 0x158) {
            // As MacroEngine::Execute reads on the back.
            const GPUVAddr source = (GPUVAddr{value_of(0x154) & 0x1ffffff} << 32) | value_of(0x155);
            const u32 words = std::min<u32>(value, 0x1000);
            const size_t read_bytes = size_t{words} * sizeof(u32);
            split.Run(
                [this, source, read_bytes] {
                    VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, source,
                                                        read_bytes);
                    if (memory_manager.IsMemoryDirty(source, read_bytes,
                                                     VideoCommon::CacheType::BufferCache)) {
                        memory_manager.FlushRegion(source, read_bytes,
                                                   VideoCommon::CacheType::BufferCache);
                    }
                },
                VideoCommon::DrainSite::FrontVerify);
            for (u32 i = 0; i < words; ++i) {
                fifo.push_back(memory_manager.Read<u32>(source + u64{i} * sizeof(u32)));
            }
        }
        return true;
    };
    auto& engine = GetNativeMacroEngine();
#ifdef ARCHITECTURE_x86_64
    if (!engine.native_jit) {
        engine.native_jit = std::make_shared<TuringMacroJit>();
    }
    engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
    engine.native_jit->Execute(start, fifo, std::span<u32>{native_data_ram}, read, emit);
#else
    engine.native_interpreter.Execute(engine.native_code, start, fifo, read, emit);
#endif
    return writes;
}

// A register as the back holds it once it ran what the front sent: the front's copy, or the
// back's once it ran the last macro there that can write it. No entry sent after that macro
// writes the register (a front write takes it over first, a later macro there would be the one
// waited for), so the back's value stays until the front writes the register again.
u32 Maxwell3D::FrontRegister(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    if (fs.back_regs[method] != 0) {
        const bool waited =
            split.WaitForPosition(fs.back_regs[method], VideoCommon::DrainSite::BackRegister);
        if (BackMacroChecks() != 0) [[unlikely]] {
            ++BackStats().reads;
            BackStats().read_waits += waited ? 1 : 0;
        }
        fs.regs[method] = regs.reg_array[method];
        fs.back_regs[method] = 0;
        fs.back_owned[method] = fs.back_shadow[method] != 0;
    }
    return fs.regs[method];
}

// Before the front writes `method`: the register, and its shadow RAM entry unless the mode
// leaves that alone, are the front's again; a replay takes the entry from the back first (as
// FrontRegister). Returns whether the back may hold another value or entry than the front's
// copies, so that the write must be sent even if it seems to change nothing.
bool Maxwell3D::FrontTakeOver(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    using Control = Regs::ShadowRamControl;
    const bool tracked = fs.control == Control::Track || fs.control == Control::TrackWithFilter;
    if (fs.back_shadow[method] != 0 && fs.control == Control::Replay) {
        const bool waited =
            split.WaitForPosition(fs.back_shadow[method], VideoCommon::DrainSite::BackRegister);
        if (BackMacroChecks() != 0) [[unlikely]] {
            ++BackStats().replays;
            BackStats().replay_waits += waited ? 1 : 0;
        }
        fs.shadow[method] = shadow_state.reg_array[method];
        fs.back_shadow[method] = 0;
    }
    const bool differs = fs.back_regs[method] != 0 || (tracked && fs.back_shadow[method] != 0);
    fs.back_regs[method] = 0;
    if (tracked) {
        fs.back_shadow[method] = 0;
    }
    fs.back_owned[method] = fs.back_shadow[method] != 0;
    return differs;
}

void Maxwell3D::FrontTrackInstanceLoop(FrontState& fs, u32 method, u32 value) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        // Instance id, bits 27:26: First leaves an instance loop, Subsequent enters one.
        if (((value >> 26) & 3) == 0) {
            fs.instance_loop = false;
        } else if (((value >> 26) & 3) == 1) {
            fs.instance_loop = true;
        }
        break;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        fs.instance_loop = false; // DrawMode::InlineIndex: draw.end draws
        break;
    default:
        break;
    }
}

bool Maxwell3D::FrontSkipsLoopWrite(FrontState& fs, u32 method, u32 value, bool shadow_kept) {
    FrontTrackInstanceLoop(fs, method, value);
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
        break;
    default:
        return false;
    }
    // Sent while state writes wait in the back's sink: it consumes them at the next executable
    // method, and a changed one ends a deferred draw that must issue before the next engine's work
    // (a compute dispatch after a dropped vertex_buffer.count would otherwise run first).
    if (!fs.skip_same || fs.sink_pending || !shadow_kept || fs.regs[method] != value) {
        return false;
    }
    // Instance loop methods flush no deferred draw, and the draw manager ignores these: DrawEnd
    // draws nothing in an instance loop, and it only reads the vertex buffer first and count
    // (index_buffer writes also mark the index buffer dirty, so they are always sent).
    return method != MAXWELL3D_REG_INDEX(draw.end) || fs.instance_loop;
}

// What the back can write to guest memory running the entry just sent, `amount` words of `method`
// (GpuSplit::Writes::Noted), for the macro DMA reads after it (FrontDmaRead). The executable
// methods (IsMethodExecutable) not named here change registers and host state only: bindings,
// barriers, counter resets, the WFI's host query copies, shadow RAM and macro code.
void Maxwell3D::FrontNoteWrites(VideoCommon::GpuSplit& split, const FrontState& fs, u32 method,
                                u32 amount) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto address = [&fs](u32 high) {
        return (GPUVAddr{fs.regs[high]} << 32) | fs.regs[high + 1];
    };
    if (fs.draw_pending) {
        // A draw, or one an instance loop deferred to this entry (ProcessDirtyRegisters): where
        // its shaders store, and what the caches download for it, only the back knows. Guest
        // memory it writes through the caches only.
        split.NoteCacheWrite();
    }
    if (method >= cb_data && method < cb_data + 16) {
        // FrontWrite and FrontCallMultiMethod advanced the offset past these words already.
        const u32 bytes = amount * static_cast<u32>(sizeof(u32));
        const u32 offset = fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] - bytes;
        split.NoteWrite(memory_manager,
                        address(MAXWELL3D_REG_INDEX(const_buffer.address_high)) + offset, bytes);
        return;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        // CounterReport: the value and its timestamp, now or once the host result is in.
        split.NoteWrite(memory_manager, address(MAXWELL3D_REG_INDEX(report_semaphore.address_high)),
                        16);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        // Upload::State::ProcessData: lines of line_length_in bytes, pitch apart, or swizzled
        // into the destination surface.
        const auto launch =
            std::bit_cast<Regs::LaunchDMA>(fs.regs[MAXWELL3D_REG_INDEX(launch_dma)]);
        Upload::Registers upload;
        std::memcpy(&upload, fs.regs.data() + MAXWELL3D_REG_INDEX(upload), sizeof(upload));
        const auto range =
            Upload::UploadedRange(upload, launch.memory_layout == Regs::LaunchDMA::Layout::Pitch);
        split.NoteWrite(memory_manager, range.address, range.size);
        break;
    }
    case MAXWELL3D_REG_INDEX(render_enable.mode): {
        // A condition read from memory (ProcessQueryCondition) flushes every cache over it (also
        // under an override a macro the back ran may have set).
        constexpr u32 override_method = MAXWELL3D_REG_INDEX(render_enable_override);
        const auto mode =
            static_cast<Regs::RenderEnable::Mode>(fs.regs[MAXWELL3D_REG_INDEX(render_enable.mode)]);
        if ((fs.back_regs[override_method] != 0 ||
             static_cast<Regs::RenderEnable::Override>(fs.regs[override_method]) ==
                 Regs::RenderEnable::Override::UseRenderEnable) &&
            (mode == Regs::RenderEnable::Mode::Conditional ||
             mode == Regs::RenderEnable::Mode::IfEqual ||
             mode == Regs::RenderEnable::Mode::IfNotEqual)) {
            split.NoteCacheWrite();
        }
        break;
    }
    case MAXWELL3D_REG_INDEX(sync_info):
        // A syncpoint increment: macros read syncpoints through memory too (MemoryManager::Read
        // of the syncpoint shim), guest memory holds nothing new.
        split.NoteCacheWrite();
        break;
    default:
        break;
    }
}

void Maxwell3D::FrontDmaRead(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                             u32 method, u32 count) {
    const GPUVAddr source = (GPUVAddr{FrontRegister(split, fs, 0x154) & 0x1ffffff} << 32) |
                            FrontRegister(split, fs, 0x155);
    const u32 ram = FrontRegister(split, fs, 0x156);
    const u32 words = std::min<u32>(count, 0x1000);
    const size_t read_bytes = size_t{words} * sizeof(u32);
    std::vector<u32> diag_before;
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        diag_before.resize(words);
        memory_manager.ReadBlockUnsafe(source, diag_before.data(), read_bytes);
    }
    VideoCommon::Nx2Profile::CountDmaRead(macro);
    if (VideoCommon::Nx2Profile::Enabled()) {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadBytes, read_bytes);
    }
    // Drains the split, and has the back write out what its caches hold of the words: data the
    // GPU wrote (indirect draw arguments from a culling dispatch), reports the query cache holds
    // until their fence or host result (else the read returns stale occlusion counts).
    const auto write_out = [&] {
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
        if (VideoCommon::QueryDma::TraceEnabled()) {
            split.Run(
                [this, macro, method, source, read_bytes] {
                    if (VideoCommon::QueryDma::HasPendingQueries(memory_manager, *rasterizer,
                                                                 source, read_bytes)) {
                        VideoCommon::QueryDma::Trace(memory_manager, *rasterizer, macro, method,
                                                     source, read_bytes);
                    }
                },
                VideoCommon::DrainSite::MacroDmaFlush);
        }
        if (memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache)) {
            split.Run(
                [memory = &memory_manager, source, read_bytes] {
                    memory->FlushRegion(source, read_bytes, VideoCommon::CacheType::BufferCache);
                },
                VideoCommon::DrainSite::MacroDmaFlush);
        }
        split.Run(
            [this, source, read_bytes] {
                VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, source,
                                                    read_bytes);
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    };
    // Only the work still queued that can write the words has to run first (FrontNoteWrites,
    // GpuSplit::WaitForWrites). Past it no entry can make them GPU-written or add reports there,
    // so the caches have nothing to write out unless they already hold it now.
    const bool waited = split.WaitForWrites(memory_manager, source, read_bytes);
    const bool drained =
        memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache) ||
        VideoCommon::QueryDma::HoldsReports(memory_manager, *rasterizer, source, read_bytes);
    if (drained) {
        write_out();
    }
    std::vector<u32> undrained;
    std::vector<u32> checked;
    if (dma_read_verify) [[unlikely]] {
        if (!drained) {
            for (u32 i = 0; i < words; ++i) {
                undrained.push_back(memory_manager.Read<u32>(source + u64{i} * sizeof(u32)));
            }
            write_out();
        }
        checked.reserve(words);
    }
    u32 changed{};
    for (u32 i = 0; i < words; ++i) {
        const u32 word = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        if (method == 0x158) {
            fs.parameters.push_back(word);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = word;
        }
        if (!diag_before.empty() && diag_before[i] != word) {
            ++changed;
        }
        if (dma_read_verify) [[unlikely]] {
            checked.push_back(word);
        }
    }
    if (dma_read_verify) [[unlikely]] {
        CheckDmaRead(macro, source, drained, waited, undrained, checked);
    }
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        bool gpu_written = false;
        for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + read_bytes; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            gpu_written |= device && Tegra::Nx2IsGpuWritten(*device);
        }
        Tegra::Nx2CmdPageDiag::RecordDmaRead(source, words, changed, gpu_written);
    }
}

void Maxwell3D::FrontVerify(VideoCommon::GpuSplit& split, FrontState& fs) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    static u64 checks{};
    static u64 differences{};
    ++checks;
    // State writes the back still holds in its method sink count as done.
    std::array<u32, Regs::NUM_REGS> expected;
    std::memcpy(expected.data(), regs.reg_array.data(), sizeof(expected));
    for (const auto& [method, value] : method_sink) {
        expected[method] = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay
                               ? shadow_state.reg_array[method]
                               : value;
    }
    // What macros the back ran may have written, the front takes over (FrontRunOnBack).
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.back_regs[i] != 0) {
            fs.regs[i] = expected[i];
        }
        if (fs.back_shadow[i] != 0) {
            fs.shadow[i] = shadow_state.reg_array[i];
        }
    }
    fs.back_regs.fill(0);
    fs.back_shadow.fill(0);
    fs.back_owned.reset();
    static u64 control_differences{};
    if ((fs.control != shadow_state.shadow_ram_control ||
         fs.shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] != shadow_state.reg_array[MAXWELL3D_REG_INDEX(shadow_ram_control)] ||
         fs.regs[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected[MAXWELL3D_REG_INDEX(shadow_ram_control)]) &&
        control_differences < 4) {
        LOG_WARNING(HW_GPU, "Split shadow RAM control writes, front ({} total) / back ({} total):",
                    fs.control_count, back_control_count);
        for (u32 i = 0; i < 16; ++i) {
            const auto& f = fs.control_writes[(fs.control_count + i) % 16];
            const auto& b = back_control_writes[(back_control_count + i) % 16];
            LOG_WARNING(HW_GPU, "  front {:#x} (mode {}, path {}) | back {:#x} (mode {}, last {})",
                        f.argument, f.before, f.path, b.argument, b.before, b.path);
        }
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.regs[i] != expected[i]) {
            if (differences++ < 64) {
                LOG_WARNING(HW_GPU, "Split front register {:#x}: front {:#x}, back {:#x}", i,
                            fs.regs[i], expected[i]);
            }
            fs.regs[i] = expected[i];
        }
    }
    if (fs.control != shadow_state.shadow_ram_control) {
        if (control_differences++ < 16) {
            LOG_WARNING(HW_GPU, "Split front shadow RAM control {} vs back {}",
                        static_cast<u32>(fs.control),
                        static_cast<u32>(shadow_state.shadow_ram_control));
        }
        fs.control = shadow_state.shadow_ram_control;
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.shadow[i] != shadow_state.reg_array[i]) {
            if (control_differences++ < 16) {
                LOG_WARNING(HW_GPU, "Split front shadow register {:#x}: front {:#x}, back {:#x}", i,
                            fs.shadow[i], shadow_state.reg_array[i]);
            }
            fs.shadow[i] = shadow_state.reg_array[i];
        }
    }
    if ((checks & 63) == 1) {
        LOG_INFO(HW_GPU, "Split front register checks: {} done, {} differences", checks, differences);
    }
}

void Maxwell3D::FinishFrontMacro(Core::System& system) {
    const VideoCommon::Nx2Profile::FineScope profile_scope{VideoCommon::Nx2Profile::Zone::MacroEnd};
    // What ProcessMacro and CallMacroMethod do around the macro.
    ConsumeSink(system);
    draw_manager.DrawDeferred(*this);
}

} // namespace Tegra::Engines
