// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <utility>
#include <vector>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>

#include <boost/container/small_vector.hpp>

namespace Shader::Backend::SPIRV {
namespace {
/// Output variable and the unscaled value it held before ScaleNoPerspectiveOutputs.
using ScaledOutputs = std::vector<std::pair<Id, Id>>;
void ConvertDepthMode(EmitContext& ctx) {
    const Id type{ctx.F32[1]};
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id screen_depth{ctx.OpFMul(type, ctx.OpFAdd(type, z, w), ctx.Constant(type, 0.5f))};
    const Id vector{ctx.OpCompositeInsert(ctx.F32[4], screen_depth, position, 2u)};
    ctx.OpStore(ctx.output_position, vector);
}

/// Write screen-linear generic outputs times Position.w (see RuntimeInfo::noperspective_outputs).
/// Returns the unscaled values so a geometry shader can restore them after EmitVertex.
ScaledOutputs ScaleNoPerspectiveOutputs(EmitContext& ctx) {
    ScaledOutputs saved;
    if (ctx.runtime_info.noperspective_outputs.none() || !Sirit::ValidId(ctx.output_position)) {
        return saved;
    }
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id w{ctx.OpCompositeExtract(ctx.F32[1], position, 3u)};
    for (size_t index = 0; index < ctx.output_generics.size(); ++index) {
        if (!ctx.runtime_info.noperspective_outputs.test(index)) {
            continue;
        }
        Id last{};
        for (const GenericElementInfo& element : ctx.output_generics[index]) {
            if (!Sirit::ValidId(element.id) || element.id.value == last.value ||
                element.num_components == 0) {
                continue;
            }
            last = element.id;
            const Id type{ctx.F32[element.num_components]};
            const Id value{ctx.OpLoad(type, element.id)};
            const Id scaled{element.num_components == 1 ? ctx.OpFMul(type, value, w)
                                                        : ctx.OpVectorTimesScalar(type, value, w)};
            ctx.OpStore(element.id, scaled);
            saved.emplace_back(element.id, value);
        }
    }
    return saved;
}

/// Geometry shaders keep output values across EmitVertex, so an output that is written once and
/// emitted several times must not be scaled again: put the unscaled values back.
void RestoreNoPerspectiveOutputs(EmitContext& ctx, const ScaledOutputs& saved) {
    for (const auto& [id, value] : saved) {
        ctx.OpStore(id, value);
    }
}

void SetFixedPipelinePointSize(EmitContext& ctx) {
    if (ctx.runtime_info.fixed_state_point_size) {
        const float point_size{*ctx.runtime_info.fixed_state_point_size};
        ctx.OpStore(ctx.output_point_size, ctx.Const(point_size));
    }
}

Id DefaultVarying(EmitContext& ctx, u32 num_components, u32 element, Id zero, Id one,
                  Id default_vector) {
    switch (num_components) {
    case 1:
        return element == 3 ? one : zero;
    case 2:
        return ctx.ConstantComposite(ctx.F32[2], zero, element + 1 == 3 ? one : zero);
    case 3:
        return ctx.ConstantComposite(ctx.F32[3], zero, zero, element + 2 == 3 ? one : zero);
    case 4:
        return default_vector;
    }
    throw InvalidArgument("Bad element");
}

Id ComparisonFunction(EmitContext& ctx, CompareFunction comparison, Id operand_1, Id operand_2) {
    switch (comparison) {
    case CompareFunction::Never:
        return ctx.false_value;
    case CompareFunction::Less:
        return ctx.OpFOrdLessThan(ctx.U1, operand_1, operand_2);
    case CompareFunction::Equal:
        return ctx.OpFOrdEqual(ctx.U1, operand_1, operand_2);
    case CompareFunction::LessThanEqual:
        return ctx.OpFOrdLessThanEqual(ctx.U1, operand_1, operand_2);
    case CompareFunction::Greater:
        return ctx.OpFOrdGreaterThan(ctx.U1, operand_1, operand_2);
    case CompareFunction::NotEqual:
        return ctx.OpFOrdNotEqual(ctx.U1, operand_1, operand_2);
    case CompareFunction::GreaterThanEqual:
        return ctx.OpFOrdGreaterThanEqual(ctx.U1, operand_1, operand_2);
    case CompareFunction::Always:
        return ctx.true_value;
    }
    throw InvalidArgument("Comparison function {}", comparison);
}

void AlphaTest(EmitContext& ctx) {
    if (!ctx.runtime_info.alpha_test_func) {
        return;
    }
    const auto comparison{*ctx.runtime_info.alpha_test_func};
    if (comparison == CompareFunction::Always) {
        return;
    }
    if (!Sirit::ValidId(ctx.frag_color[0])) {
        return;
    }

    const Id type{ctx.F32[1]};
    const Id rt0_color{ctx.OpLoad(ctx.F32[4], ctx.frag_color[0])};
    const Id alpha{ctx.OpCompositeExtract(type, rt0_color, 3u)};

    const Id true_label{ctx.OpLabel()};
    const Id discard_label{ctx.OpLabel()};
    const Id alpha_reference{ctx.Const(ctx.runtime_info.alpha_test_reference)};
    const Id condition{ComparisonFunction(ctx, comparison, alpha, alpha_reference)};

    ctx.OpSelectionMerge(true_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, true_label, discard_label);
    ctx.AddLabel(discard_label);
    ctx.OpKill();
    ctx.AddLabel(true_label);
}

Id OotMotionWord(EmitContext& ctx, u32 word) {
    const Id pointer_type{ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1])};
    const Id pointer{ctx.OpAccessChain(pointer_type, ctx.oot_motion.push_constant,
                                       ctx.Const(ctx.oot_motion.member_index + word))};
    return ctx.OpLoad(ctx.U32[1], pointer);
}

Id OotMotionEnabled(EmitContext& ctx) {
    const Id flags{OotMotionWord(ctx, offsetof(OotMotionLayout, flags) / 4)};
    return ctx.OpINotEqual(ctx.U1, ctx.OpBitwiseAnd(ctx.U32[1], flags, ctx.Const(1u)),
                           ctx.u32_zero_value);
}

// Stores this vertex's clip position in its history slot and passes it and last frame's to the
// fragment stage. A previous position with w = 0 means there is none.
void OotMotionVertex(EmitContext& ctx) {
    if (!Sirit::ValidId(ctx.oot_motion.history)) {
        return;
    }
    const Id u32_type{ctx.U32[1]};
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    ctx.OpStore(ctx.oot_motion.current, position);
    ctx.OpStore(ctx.oot_motion.previous,
                ctx.OpCompositeInsert(ctx.F32[4], ctx.f32_zero_value, position, 3u));

    const Id current_base{OotMotionWord(ctx, offsetof(OotMotionLayout, current_base) / 4)};
    const Id previous_base{OotMotionWord(ctx, offsetof(OotMotionLayout, previous_base) / 4)};
    const Id vertex_bias{OotMotionWord(ctx, offsetof(OotMotionLayout, vertex_bias) / 4)};
    const Id range{OotMotionWord(ctx, offsetof(OotMotionLayout, vertex_range) / 4)};
    const Id instance_bias{OotMotionWord(ctx, offsetof(OotMotionLayout, instance_bias) / 4)};
    const Id local{ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.oot_motion.vertex_index),
                              vertex_bias)};
    const Id instance{ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.oot_motion.instance_index),
                                 instance_bias)};
    const Id condition{ctx.OpLogicalAnd(ctx.U1, OotMotionEnabled(ctx),
                                        ctx.OpULessThan(ctx.U1, local, range))};
    const Id store_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, store_label, merge_label);
    ctx.AddLabel(store_label);
    const Id slot{ctx.OpIAdd(u32_type, ctx.OpIMul(u32_type, instance, range), local)};
    const Id current_index{ctx.OpIAdd(u32_type, current_base, slot)};
    const Id has_previous{ctx.OpINotEqual(ctx.U1, previous_base, ctx.Const(~0u))};
    const Id previous_index{ctx.OpSelect(u32_type, has_previous,
                                         ctx.OpIAdd(u32_type, previous_base, slot),
                                         current_index)};
    const Id previous{ctx.OpLoad(
        ctx.F32[4], ctx.OpAccessChain(ctx.oot_motion.history_element, ctx.oot_motion.history,
                                      ctx.u32_zero_value, previous_index))};
    ctx.OpStore(ctx.OpAccessChain(ctx.oot_motion.history_element, ctx.oot_motion.history,
                                  ctx.u32_zero_value, current_index),
                position);
    const Id previous_w{ctx.OpSelect(ctx.F32[1], has_previous,
                                     ctx.OpCompositeExtract(ctx.F32[1], previous, 3u),
                                     ctx.f32_zero_value)};
    ctx.OpStore(ctx.oot_motion.previous,
                ctx.OpCompositeInsert(ctx.F32[4], previous_w, previous, 3u));
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

} // Anonymous namespace

// Mesh stage: as OotMotionVertex for one vertex of the workgroup. Each vertex has two history
// entries: its clip position and a signature of its other attributes (texture coordinates and
// such). Mesh programs that cull and compact their output vertices can put another vertex at
// the same index next frame; the history is used only when the signature matches. The slot is
// the workgroup's index in the draw (vertex_range holds the draw's workgroup count) times the
// workgroup's vertex count, plus the vertex.
void EmitOotMotionMeshVertex(EmitContext& ctx, Id vertex) {
    const Id u32_type{ctx.U32[1]};
    const Id vec4_output{ctx.TypePointer(spv::StorageClass::Output, ctx.F32[4])};
    const Id position{
        ctx.OpLoad(ctx.F32[4], ctx.OpAccessChain(vec4_output, ctx.output_position, vertex))};
    const Id current_pointer{ctx.OpAccessChain(vec4_output, ctx.oot_motion.current, vertex)};
    const Id previous_pointer{ctx.OpAccessChain(vec4_output, ctx.oot_motion.previous, vertex)};
    ctx.OpStore(current_pointer, position);
    ctx.OpStore(previous_pointer,
                ctx.OpCompositeInsert(ctx.F32[4], ctx.f32_zero_value, position, 3u));

    // Signature: FNV-1a over the vertex's non-position attribute words in the stage buffer
    // region (line 0 of the vertex at base + (vertex / 32) * lines * 32 + vertex % 32).
    const auto& lines{ctx.mesh_info.vertex_lines};
    const u32 num_lines{static_cast<u32>(lines.size())};
    const Id group{ctx.OpShiftRightLogical(u32_type, vertex, ctx.Const(5U))};
    const Id lane{ctx.OpBitwiseAnd(u32_type, vertex, ctx.Const(31U))};
    const Id first_word{
        ctx.OpIAdd(u32_type, ctx.OpIMul(u32_type, group, ctx.Const(num_lines * 32)), lane)};
    // EDEN_OOT_MESH_SIGNATURE=first|last: only the first or last such word (diagnosis).
    static const char signature_lines = [] {
        const char* value = std::getenv("EDEN_OOT_MESH_SIGNATURE");
        return value != nullptr ? value[0] : 'a';
    }();
    boost::container::small_vector<u32, 32> hashed;
    for (u32 line = 0; line < num_lines; ++line) {
        const u32 attribute{lines[line]};
        if (attribute < static_cast<u32>(IR::Attribute::PositionX) ||
            attribute > static_cast<u32>(IR::Attribute::PositionW)) {
            hashed.push_back(line);
        }
    }
    if (!hashed.empty() && signature_lines == 'f') {
        hashed.resize(1);
    } else if (!hashed.empty() && signature_lines == 'l') {
        hashed.erase(hashed.begin(), hashed.end() - 1);
    }
    Id signature{ctx.Const(2166136261U)};
    for (const u32 line : hashed) {
        const Id word{ctx.OpLoad(
            u32_type, ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.mesh_attributes,
                                        ctx.OpIAdd(u32_type, first_word, ctx.Const(line * 32))))};
        signature = ctx.OpIMul(u32_type, ctx.OpBitwiseXor(u32_type, signature, word),
                               ctx.Const(16777619U));
    }
    const Id signature_vector{ctx.OpCompositeConstruct(
        ctx.F32[4], ctx.OpBitcast(ctx.F32[1], signature), ctx.f32_zero_value, ctx.f32_zero_value,
        ctx.f32_zero_value)};

    const Id current_base{OotMotionWord(ctx, offsetof(OotMotionLayout, current_base) / 4)};
    const Id previous_base{OotMotionWord(ctx, offsetof(OotMotionLayout, previous_base) / 4)};
    const Id workgroups{OotMotionWord(ctx, offsetof(OotMotionLayout, vertex_range) / 4)};
    const Id workgroup{
        ctx.OpCompositeExtract(u32_type, ctx.OpLoad(ctx.U32[3], ctx.workgroup_id), 0U)};
    const Id condition{ctx.OpLogicalAnd(ctx.U1, OotMotionEnabled(ctx),
                                        ctx.OpULessThan(ctx.U1, workgroup, workgroups))};
    const Id store_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, store_label, merge_label);
    ctx.AddLabel(store_label);
    // Two entries per vertex: position, then signature.
    const Id slot{ctx.OpShiftLeftLogical(
        u32_type,
        ctx.OpIAdd(u32_type,
                   ctx.OpIMul(u32_type, workgroup, ctx.Const(ctx.mesh_info.max_vertices)), vertex),
        ctx.Const(1U))};
    const auto entry = [&](Id base, Id offset) {
        return ctx.OpAccessChain(ctx.oot_motion.history_element, ctx.oot_motion.history,
                                 ctx.u32_zero_value,
                                 ctx.OpIAdd(u32_type, ctx.OpIAdd(u32_type, base, slot), offset));
    };
    const Id has_previous{ctx.OpINotEqual(ctx.U1, previous_base, ctx.Const(~0u))};
    const Id read_base{ctx.OpSelect(u32_type, has_previous, previous_base, current_base)};
    const Id previous{ctx.OpLoad(ctx.F32[4], entry(read_base, ctx.u32_zero_value))};
    const Id previous_signature{ctx.OpBitcast(
        u32_type, ctx.OpCompositeExtract(ctx.F32[1],
                                         ctx.OpLoad(ctx.F32[4], entry(read_base, ctx.Const(1U))),
                                         0u))};
    ctx.OpStore(entry(current_base, ctx.u32_zero_value), position);
    ctx.OpStore(entry(current_base, ctx.Const(1U)), signature_vector);
    const Id same_vertex{ctx.OpLogicalAnd(ctx.U1, has_previous,
                                          ctx.OpIEqual(ctx.U1, previous_signature, signature))};
    const Id previous_w{ctx.OpSelect(ctx.F32[1], same_vertex,
                                     ctx.OpCompositeExtract(ctx.F32[1], previous, 3u),
                                     ctx.f32_zero_value)};
    ctx.OpStore(previous_pointer, ctx.OpCompositeInsert(ctx.F32[4], previous_w, previous, 3u));
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

namespace {
// The history word `word` words after the slot `base` (four words per slot).
Id OotHistoryWord(EmitContext& ctx, Id base, Id word) {
    const Id u32_type{ctx.U32[1]};
    const Id pointer{ctx.TypePointer(spv::StorageClass::StorageBuffer, ctx.F32[1])};
    return ctx.OpAccessChain(
        pointer, ctx.oot_motion.history, ctx.u32_zero_value,
        ctx.OpIAdd(u32_type, base, ctx.OpShiftRightLogical(u32_type, word, ctx.Const(2U))),
        ctx.OpBitwiseAnd(u32_type, word, ctx.Const(3U)));
}
} // Anonymous namespace

// Last frame's word of a constant buffer for the shadow evaluation (OotShadowPass): from the
// snapshot at previous_base, or this frame's when there is none.
Id EmitOotPrevCbufU32(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset) {
    const Id current{EmitGetCbufU32(ctx, binding, offset)};
    if (!ctx.oot_motion.shadow || !binding.IsImmediate() ||
        binding.U32() >= ctx.oot_motion.shadow_size.size() ||
        ctx.oot_motion.shadow_size[binding.U32()] == 0) {
        return current;
    }
    const Id u32_type{ctx.U32[1]};
    const u32 size{ctx.oot_motion.shadow_size[binding.U32()]};
    const Id word{offset.IsImmediate() ? ctx.Const(offset.U32() / 4)
                                       : ctx.OpShiftRightLogical(u32_type, ctx.Def(offset),
                                                                 ctx.Const(2U))};
    const Id index{ctx.OpIAdd(u32_type, ctx.Const(ctx.oot_motion.shadow_base[binding.U32()]),
                              ctx.OpUMin(u32_type, word, ctx.Const(size - 1)))};
    const Id previous_base{OotMotionWord(ctx, offsetof(OotMotionLayout, previous_base) / 4)};
    const Id has_previous{ctx.OpLogicalAnd(
        ctx.U1,
        ctx.OpLogicalAnd(ctx.U1, OotMotionEnabled(ctx),
                         ctx.OpINotEqual(ctx.U1, previous_base, ctx.Const(~0u))),
        ctx.OpULessThan(ctx.U1, word, ctx.Const(size)))};
    const Id base{ctx.OpSelect(u32_type, has_previous, previous_base, ctx.u32_zero_value)};
    const Id value{
        ctx.OpBitcast(u32_type, ctx.OpLoad(ctx.F32[1], OotHistoryWord(ctx, base, index)))};
    return ctx.OpSelect(u32_type, has_previous, value, current);
}

// The shadow value of an attribute write: kept when it is a vertex's position component.
void EmitOotShadowWriteAttribute(EmitContext& ctx, Id address, Id value) {
    if (!ctx.oot_motion.shadow || ctx.stage != Stage::Mesh) {
        return;
    }
    const auto& lines{ctx.mesh_info.vertex_lines};
    const u32 num_lines{static_cast<u32>(lines.size())};
    if (num_lines == 0) {
        return;
    }
    const Id u32_type{ctx.U32[1]};
    const Id index{ctx.OpShiftRightLogical(u32_type, address, ctx.Const(2U))};
    const Id group_words{ctx.Const(num_lines * 32)};
    const Id rest{ctx.OpUMod(u32_type, index, group_words)};
    const Id line{ctx.OpShiftRightLogical(u32_type, rest, ctx.Const(5U))};
    const Id vertex{ctx.OpIAdd(
        u32_type, ctx.OpIMul(u32_type, ctx.OpUDiv(u32_type, index, group_words), ctx.Const(32U)),
        ctx.OpBitwiseAnd(u32_type, rest, ctx.Const(31U)))};
    Id component{ctx.Const(4U)};
    for (u32 element = 0; element < 4; ++element) {
        const u32 attribute{static_cast<u32>(IR::Attribute::PositionX) + element};
        const auto found{std::ranges::find(lines, attribute)};
        if (found == lines.end()) {
            continue;
        }
        const u32 position_line{static_cast<u32>(found - lines.begin())};
        component = ctx.OpSelect(u32_type, ctx.OpIEqual(ctx.U1, line, ctx.Const(position_line)),
                                 ctx.Const(element), component);
    }
    const u32 spare{ctx.mesh_info.max_vertices * 4};
    const Id keep{ctx.OpLogicalAnd(
        ctx.U1, ctx.OpULessThan(ctx.U1, index, ctx.Const(ctx.mesh_primitive_base)),
        ctx.OpLogicalAnd(ctx.U1, ctx.OpULessThan(ctx.U1, component, ctx.Const(4U)),
                         ctx.OpULessThan(ctx.U1, vertex, ctx.Const(ctx.mesh_info.max_vertices))))};
    const Id target{ctx.OpSelect(
        u32_type, keep,
        ctx.OpIAdd(u32_type, ctx.OpShiftLeftLogical(u32_type, vertex, ctx.Const(2U)), component),
        ctx.Const(spare))};
    ctx.OpStore(ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.oot_motion.shadow_positions, target),
                value);
}

// Mesh epilogue with the shadow evaluation: workgroup 0 keeps this frame's constant buffer words
// the shadow reads, for next frame's shadow.
void EmitOotMotionMeshSnapshot(EmitContext& ctx) {
    const Id u32_type{ctx.U32[1]};
    const Id workgroup{ctx.OpLoad(ctx.U32[3], ctx.workgroup_id)};
    const auto is_zero = [&](u32 element) {
        return ctx.OpIEqual(ctx.U1, ctx.OpCompositeExtract(u32_type, workgroup, element),
                            ctx.u32_zero_value);
    };
    const Id first{
        ctx.OpLogicalAnd(ctx.U1, is_zero(0), ctx.OpLogicalAnd(ctx.U1, is_zero(1), is_zero(2)))};
    const Id condition{ctx.OpLogicalAnd(ctx.U1, OotMotionEnabled(ctx), first)};
    const Id copy_label{ctx.OpLabel()};
    const Id done_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(done_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, copy_label, done_label);
    ctx.AddLabel(copy_label);
    const Id current_base{OotMotionWord(ctx, offsetof(OotMotionLayout, current_base) / 4)};
    const Id invocation{
        ctx.OpCompositeExtract(u32_type, ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id), 0U)};
    const Id uniform_type{ctx.uniform_types.U32};
    for (u32 binding = 0; binding < ctx.oot_motion.shadow_size.size(); ++binding) {
        const u32 size{ctx.oot_motion.shadow_size[binding]};
        if (size == 0 || !Sirit::ValidId(ctx.cbufs[binding].U32)) {
            continue;
        }
        // for (i = invocation; i < size; i += threads)
        ctx.OpStore(ctx.oot_motion.counter, invocation);
        const Id header{ctx.OpLabel()};
        const Id check{ctx.OpLabel()};
        const Id body{ctx.OpLabel()};
        const Id next{ctx.OpLabel()};
        const Id merge{ctx.OpLabel()};
        ctx.OpBranch(header);
        ctx.AddLabel(header);
        ctx.OpLoopMerge(merge, next, spv::LoopControlMask::MaskNone);
        ctx.OpBranch(check);
        ctx.AddLabel(check);
        const Id i{ctx.OpLoad(u32_type, ctx.oot_motion.counter)};
        ctx.OpBranchConditional(ctx.OpULessThan(ctx.U1, i, ctx.Const(size)), body, merge);
        ctx.AddLabel(body);
        const Id value{ctx.OpLoad(
            u32_type,
            ctx.OpAccessChain(uniform_type, ctx.cbufs[binding].U32, ctx.u32_zero_value, i))};
        const Id word{ctx.OpIAdd(u32_type, i, ctx.Const(ctx.oot_motion.shadow_base[binding]))};
        ctx.OpStore(OotHistoryWord(ctx, current_base, word), ctx.OpBitcast(ctx.F32[1], value));
        ctx.OpBranch(next);
        ctx.AddLabel(next);
        ctx.OpStore(ctx.oot_motion.counter, ctx.OpIAdd(u32_type, i, ctx.Const(ctx.mesh_threads)));
        ctx.OpBranch(header);
        ctx.AddLabel(merge);
    }
    ctx.OpBranch(done_label);
    ctx.AddLabel(done_label);
}

// As EmitOotMotionMeshVertex, with the shadow evaluation's position as last frame's.
void EmitOotMotionMeshShadowVertex(EmitContext& ctx, Id vertex) {
    const Id u32_type{ctx.U32[1]};
    const Id vec4_output{ctx.TypePointer(spv::StorageClass::Output, ctx.F32[4])};
    const Id position{
        ctx.OpLoad(ctx.F32[4], ctx.OpAccessChain(vec4_output, ctx.output_position, vertex))};
    ctx.OpStore(ctx.OpAccessChain(vec4_output, ctx.oot_motion.current, vertex), position);
    const Id first{ctx.OpShiftLeftLogical(u32_type, vertex, ctx.Const(2U))};
    std::array<Id, 4> components;
    for (u32 element = 0; element < 4; ++element) {
        components[element] = ctx.OpBitcast(
            ctx.F32[1],
            ctx.OpLoad(u32_type,
                       ctx.OpAccessChain(ctx.mesh_shared_u32, ctx.oot_motion.shadow_positions,
                                         ctx.OpIAdd(u32_type, first, ctx.Const(element)))));
    }
    const Id previous_base{OotMotionWord(ctx, offsetof(OotMotionLayout, previous_base) / 4)};
    // Snapshots of repeated draws are matched by draw order: a shadow position more than 0.1
    // (NDC) from the current one is taken for another draw's and dropped.
    const auto ndc = [&](Id x, Id y, Id w) {
        return ctx.OpFDiv(ctx.F32[2], ctx.OpCompositeConstruct(ctx.F32[2], x, y),
                          ctx.OpCompositeConstruct(ctx.F32[2], w, w));
    };
    const Id current_ndc{ndc(ctx.OpCompositeExtract(ctx.F32[1], position, 0U),
                             ctx.OpCompositeExtract(ctx.F32[1], position, 1U),
                             ctx.OpCompositeExtract(ctx.F32[1], position, 3U))};
    const Id previous_ndc{ndc(components[0], components[1], components[3])};
    const Id distance{ctx.OpFSub(ctx.F32[2], previous_ndc, current_ndc)};
    const Id limit{ctx.Const(0.1f)};
    const Id near{ctx.OpLogicalAnd(
        ctx.U1,
        ctx.OpFOrdLessThan(ctx.U1, ctx.OpFAbs(ctx.F32[1], ctx.OpCompositeExtract(ctx.F32[1],
                                                                                 distance, 0U)),
                           limit),
        ctx.OpFOrdLessThan(ctx.U1, ctx.OpFAbs(ctx.F32[1], ctx.OpCompositeExtract(ctx.F32[1],
                                                                                 distance, 1U)),
                           limit))};
    const Id has_previous{ctx.OpLogicalAnd(
        ctx.U1,
        ctx.OpLogicalAnd(ctx.U1, OotMotionEnabled(ctx),
                         ctx.OpINotEqual(ctx.U1, previous_base, ctx.Const(~0u))),
        ctx.OpLogicalAnd(ctx.U1, near,
                         ctx.OpFOrdGreaterThan(ctx.U1, components[3], ctx.f32_zero_value)))};
    // w = 0: no previous position (the combine pass then keeps the camera's motion).
    const Id previous_w{ctx.OpSelect(ctx.F32[1], has_previous, components[3], ctx.f32_zero_value)};
    const Id previous{ctx.OpCompositeConstruct(ctx.F32[4], components[0], components[1],
                                               components[2], previous_w)};
    ctx.OpStore(ctx.OpAccessChain(vec4_output, ctx.oot_motion.previous, vertex), previous);
}

namespace {
// Keeps the nearest fragment's motion (half-float NDC offset to last frame, x and y) with its
// depth in the high word: an atomic minimum, since storage writes are not ordered by the depth
// test. The combine pass uses it where that depth is the scene's.
void OotMotionFragment(EmitContext& ctx) {
    if (!Sirit::ValidId(ctx.oot_motion.buffer)) {
        return;
    }
    const Id f32_type{ctx.F32[1]};
    const Id current{ctx.OpLoad(ctx.F32[4], ctx.oot_motion.current)};
    const Id previous{ctx.OpLoad(ctx.F32[4], ctx.oot_motion.previous)};
    const Id current_w{ctx.OpCompositeExtract(f32_type, current, 3u)};
    const Id previous_w{ctx.OpCompositeExtract(f32_type, previous, 3u)};
    const Id valid{ctx.OpFOrdGreaterThan(ctx.U1, ctx.OpFMul(f32_type, current_w, previous_w),
                                         ctx.Const(1e-12f))};
    const Id condition{ctx.OpLogicalAnd(ctx.U1, OotMotionEnabled(ctx), valid)};
    const Id write_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, write_label, merge_label);
    ctx.AddLabel(write_label);
    const auto ndc = [&](Id clip, Id w) {
        const Id xy{ctx.OpVectorShuffle(ctx.F32[2], clip, clip, 0u, 1u)};
        return ctx.OpFDiv(ctx.F32[2], xy, ctx.OpCompositeConstruct(ctx.F32[2], w, w));
    };
    const Id half{ctx.Const(0.5f)};
    const Id motion{ctx.OpFMul(ctx.F32[2],
                               ctx.OpFSub(ctx.F32[2], ndc(previous, previous_w),
                                          ndc(current, current_w)),
                               ctx.OpCompositeConstruct(ctx.F32[2], half, half))};
    const Id frag_coord{ctx.OpLoad(ctx.F32[4], ctx.oot_motion.frag_coord)};
    const Id depth{ctx.OpBitcast(ctx.U32[1], ctx.OpCompositeExtract(f32_type, frag_coord, 2u))};
    const Id x{ctx.OpConvertFToU(ctx.U32[1], ctx.OpCompositeExtract(f32_type, frag_coord, 0u))};
    const Id y{ctx.OpConvertFToU(ctx.U32[1], ctx.OpCompositeExtract(f32_type, frag_coord, 1u))};
    const Id width{OotMotionWord(ctx, offsetof(OotMotionLayout, image_width) / 4)};
    const Id index{ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], y, width), x)};
    const Id value{ctx.OpBitwiseOr(
        ctx.U64, ctx.OpShiftLeftLogical(ctx.U64, ctx.OpUConvert(ctx.U64, depth), ctx.Const(32u)),
        ctx.OpUConvert(ctx.U64, ctx.OpPackHalf2x16(ctx.U32[1], motion)))};
    const Id pointer{ctx.OpAccessChain(ctx.oot_motion.buffer_element, ctx.oot_motion.buffer,
                                       ctx.u32_zero_value, index)};
    ctx.OpAtomicUMin(ctx.U64, pointer, ctx.Const(static_cast<u32>(spv::Scope::Device)),
                     ctx.u32_zero_value, value);
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}
} // Anonymous namespace

void EmitPrologue(EmitContext& ctx) {
    if (ctx.stage == Stage::Fragment && ctx.runtime_info.dual_source_blend) {
        // Initialize dual-source blending outputs - prevents MoltenVK crash.
        const Id zero{ctx.Const(0.0f)};
        const Id one{ctx.Const(1.0f)};
        const Id default_color{ctx.ConstantComposite(ctx.F32[4], zero, zero, zero, one)};
        for (u32 i = 0; i < 2; ++i) {
            if (Sirit::ValidId(ctx.frag_color[i])) {
                ctx.OpStore(ctx.frag_color[i], default_color);
            }
        }
    }
    if (ctx.stage == Stage::VertexB) {
        const Id zero{ctx.Const(0.0f)};
        const Id one{ctx.Const(1.0f)};
        const Id default_vector{ctx.ConstantComposite(ctx.F32[4], zero, zero, zero, one)};
        ctx.OpStore(ctx.output_position, default_vector);
        for (const auto& info : ctx.output_generics) {
            if (info[0].num_components == 0) {
                continue;
            }
            u32 element{0};
            while (element < 4) {
                const auto& element_info{info[element]};
                const u32 num{element_info.num_components};
                const Id value{DefaultVarying(ctx, num, element, zero, one, default_vector)};
                ctx.OpStore(element_info.id, value);
                element += num;
            }
        }

        if (Sirit::ValidId(ctx.clip_distances)) {
            for (u32 i = 0; i < ctx.profile.max_user_clip_distances; ++i) {
                const Id idx = ctx.Const(i);
                const Id element = ctx.OpAccessChain(ctx.output_f32, ctx.clip_distances, idx);
                ctx.OpStore(element, ctx.Const(0.0f));
            }
        }
    }
    if (ctx.stage == Stage::VertexB || ctx.stage == Stage::Geometry) {
        SetFixedPipelinePointSize(ctx);
    }
    if (ctx.stage == Stage::Task || ctx.stage == Stage::Mesh) {
        EmitMeshPrologue(ctx);
    }
}

void EmitEpilogue(EmitContext& ctx) {
    if (ctx.stage == Stage::Task || ctx.stage == Stage::Mesh) {
        EmitMeshEpilogue(ctx);
        return;
    }
    if (ctx.stage == Stage::VertexB) {
        OotMotionVertex(ctx);
    }
    if (ctx.stage == Stage::VertexB && ctx.runtime_info.convert_depth_mode &&
        !ctx.profile.support_native_ndc) {
        ConvertDepthMode(ctx);
    }
    if (ctx.stage == Stage::VertexB || ctx.stage == Stage::TessellationEval) {
        ScaleNoPerspectiveOutputs(ctx);
    }
    if (ctx.stage == Stage::Fragment) {
        AlphaTest(ctx);
        OotMotionFragment(ctx);
    }
}

void EmitEmitVertex(EmitContext& ctx, const IR::Value& stream) {
    if (ctx.runtime_info.convert_depth_mode && !ctx.profile.support_native_ndc) {
        ConvertDepthMode(ctx);
    }
    const ScaledOutputs scaled{ScaleNoPerspectiveOutputs(ctx)};
    // Stream 0 is the plain EmitVertex, which needs no GeometryStreams capability.
    if (stream.IsImmediate() && stream.U32() == 0) {
        ctx.OpEmitVertex();
    } else if (!ctx.profile.support_geometry_streams) {
        throw NotImplementedException("Geometry streams");
    } else if (stream.IsImmediate()) {
        ctx.OpEmitStreamVertex(ctx.Def(stream));
    } else {
        LOG_WARNING(Shader_SPIRV, "Stream is not immediate");
        ctx.OpEmitStreamVertex(ctx.u32_zero_value);
    }
    // Restore fixed pipeline point size after emitting the vertex
    SetFixedPipelinePointSize(ctx);
    RestoreNoPerspectiveOutputs(ctx, scaled);
}

void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
    // Stream 0 is the plain EndPrimitive, which needs no GeometryStreams capability.
    if (stream.IsImmediate() && stream.U32() == 0) {
        ctx.OpEndPrimitive();
    } else if (!ctx.profile.support_geometry_streams) {
        throw NotImplementedException("Geometry streams");
    } else if (stream.IsImmediate()) {
        ctx.OpEndStreamPrimitive(ctx.Def(stream));
    } else {
        LOG_WARNING(Shader_SPIRV, "Stream is not immediate");
        ctx.OpEndStreamPrimitive(ctx.u32_zero_value);
    }
}

} // namespace Shader::Backend::SPIRV
