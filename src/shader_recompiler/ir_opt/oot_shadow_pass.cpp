// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ocarina of Time object motion for mesh programs: a second evaluation of everything a mesh
// program's attribute writes compute from constant buffers, with last frame's constant buffers
// (OotPrevCbufU32). Each attribute write gets an OotShadowWriteAttribute twin with the shadow
// value, so the epilogue has last frame's position of the same vertex whatever the culling did.
// The shadow follows this frame's control flow, and memory loads read this frame's data.

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/div_ceil.h"
#include "common/logging.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
// Instanced meshes (falling leaves, foliage) index whole 64 KiB constant buffers per instance,
// up to about 36,600 words in Kokiri Forest. The history holds 4M slots of four words a frame.
constexpr u32 MaxShadowWords = 65536;

u32 CbufElementSize(IR::Opcode op) {
    switch (op) {
    case IR::Opcode::GetCbufU8:
    case IR::Opcode::GetCbufS8:
        return 1;
    case IR::Opcode::GetCbufU16:
    case IR::Opcode::GetCbufS16:
        return 2;
    case IR::Opcode::GetCbufU32:
    case IR::Opcode::GetCbufF32:
        return 4;
    case IR::Opcode::GetCbufU32x2:
        return 8;
    default:
        return 0;
    }
}

bool IsShadowCbuf(const IR::Inst& inst) {
    return CbufElementSize(inst.GetOpcode()) != 0 && inst.Arg(0).IsImmediate() &&
           inst.Arg(0).U32() < Info::MAX_CBUFS;
}

// Instructions the shadow can evaluate again: pure values, phis and constant buffer reads.
bool IsShadowable(const IR::Inst& inst) {
    const IR::Opcode op = inst.GetOpcode();
    if (op == IR::Opcode::Phi || IsShadowCbuf(inst)) {
        return true;
    }
    return op != IR::Opcode::Identity && op != IR::Opcode::Void &&
           CbufElementSize(op) == 0 && !inst.MayHaveSideEffects() &&
           !inst.IsPseudoInstruction();
}

// Constant buffer reads and the shadowable instructions that depend on them.
std::unordered_set<const IR::Inst*> Tainted(const IR::Program& program) {
    std::unordered_set<const IR::Inst*> tainted;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = program.post_order_blocks.rbegin(); it != program.post_order_blocks.rend();
             ++it) {
            for (const IR::Inst& inst : (*it)->Instructions()) {
                if (tainted.contains(&inst) || !IsShadowable(inst)) {
                    continue;
                }
                bool taint = IsShadowCbuf(inst);
                for (size_t index = 0; !taint && index < inst.NumArgs(); ++index) {
                    const IR::Value arg = inst.Arg(index);
                    taint = !arg.IsImmediate() && tainted.contains(arg.InstRecursive());
                }
                if (taint) {
                    tainted.insert(&inst);
                    changed = true;
                }
            }
        }
    }
    return tainted;
}
} // Anonymous namespace

bool OotShadowPass(IR::Program& program) {
    const std::unordered_set<const IR::Inst*> tainted = Tainted(program);
    if (tainted.empty()) {
        return false;
    }
    auto& motion = program.info.oot_motion;

    // Words of each constant buffer the shadow reads: up to its furthest immediate read, or
    // what the program uses of it when an offset is dynamic.
    std::array<u32, Info::MAX_CBUFS> bytes{};
    for (const IR::Inst* inst : tainted) {
        if (!IsShadowCbuf(*inst)) {
            continue;
        }
        const u32 binding = inst->Arg(0).U32();
        const IR::Value offset = inst->Arg(1);
        const u32 end = offset.IsImmediate()
                            ? offset.U32() + CbufElementSize(inst->GetOpcode())
                            : program.info.constant_buffer_used_sizes[binding];
        bytes[binding] = (std::max)(bytes[binding], end);
    }
    u32 words = 0;
    for (u32 binding = 0; binding < Info::MAX_CBUFS; ++binding) {
        motion.shadow_base[binding] = words;
        motion.shadow_size[binding] = Common::DivCeil(bytes[binding], 4u);
        words += motion.shadow_size[binding];
    }
    if (words == 0 || words > MaxShadowWords) {
        std::string sizes;
        for (u32 binding = 0; binding < Info::MAX_CBUFS; ++binding) {
            if (bytes[binding] != 0) {
                sizes += fmt::format(" c{}:{}", binding, bytes[binding]);
            }
        }
        LOG_INFO(Shader, "OoT shadow pass: {} words of constants (limit {}):{}", words,
                 MaxShadowWords, sizes);
        return false;
    }
    motion.shadow_words = words;

    std::unordered_map<const IR::Inst*, IR::Value> shadow;
    const auto shadow_of = [&](const IR::Value& value) {
        if (value.IsImmediate()) {
            return value;
        }
        const auto it = shadow.find(value.InstRecursive());
        return it != shadow.end() ? it->second : value;
    };
    std::vector<IR::Inst*> phis;
    for (auto block_it = program.post_order_blocks.rbegin();
         block_it != program.post_order_blocks.rend(); ++block_it) {
        IR::Block& block = **block_it;
        for (auto it = block.begin(); it != block.end(); ++it) {
            IR::Inst& inst = *it;
            if (!tainted.contains(&inst)) {
                continue;
            }
            const auto next = std::next(it);
            if (inst.GetOpcode() == IR::Opcode::Phi) {
                // Operands once every block has its shadows (back edges come later).
                IR::Inst& phi = *block.PrependNewInst(block.begin(), IR::Opcode::Phi, {},
                                                      inst.Flags<u32>());
                shadow.emplace(&inst, IR::Value{&phi});
                phis.push_back(&inst);
                continue;
            }
            if (IsShadowCbuf(inst)) {
                IR::IREmitter ir{block, next};
                const IR::U32 binding{inst.Arg(0)};
                const IR::U32 offset{inst.Arg(1)};
                const auto word = [&](const IR::U32& byte_offset) {
                    return IR::U32{IR::Value{&*block.PrependNewInst(
                        next, IR::Opcode::OotPrevCbufU32, {binding, byte_offset})}};
                };
                IR::Value value;
                switch (inst.GetOpcode()) {
                case IR::Opcode::GetCbufU32:
                    value = word(offset);
                    break;
                case IR::Opcode::GetCbufF32:
                    value = ir.BitCast<IR::F32>(word(offset));
                    break;
                case IR::Opcode::GetCbufU32x2:
                    value = ir.CompositeConstruct(word(offset),
                                                  word(IR::U32{ir.IAdd(offset, ir.Imm32(4))}));
                    break;
                default: {
                    const bool is_signed = inst.GetOpcode() == IR::Opcode::GetCbufS8 ||
                                           inst.GetOpcode() == IR::Opcode::GetCbufS16;
                    const u32 bits = CbufElementSize(inst.GetOpcode()) * 8;
                    const IR::U32 shift{ir.ShiftLeftLogical(ir.BitwiseAnd(offset, ir.Imm32(3)),
                                                            ir.Imm32(3))};
                    value = ir.BitFieldExtract(word(offset), shift, ir.Imm32(bits), is_signed);
                    break;
                }
                }
                shadow.emplace(&inst, value);
                it = std::prev(next);
                continue;
            }
            IR::Inst& copy = *block.PrependNewInst(next, inst);
            for (size_t index = 0; index < copy.NumArgs(); ++index) {
                const IR::Value arg = copy.Arg(index);
                if (const IR::Value replaced = shadow_of(arg); replaced != arg) {
                    copy.SetArg(index, replaced);
                }
            }
            shadow.emplace(&inst, IR::Value{&copy});
            it = std::prev(next);
        }
    }
    for (IR::Inst* phi : phis) {
        IR::Inst* const copy = shadow.at(phi).InstRecursive();
        for (size_t index = 0; index < phi->NumArgs(); ++index) {
            copy->AddPhiOperand(phi->PhiBlock(index), shadow_of(phi->Arg(index)));
        }
    }
    for (IR::Block* const block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            if (it->GetOpcode() != IR::Opcode::IsbeWriteAttribute) {
                continue;
            }
            it = block->PrependNewInst(std::next(it), IR::Opcode::OotShadowWriteAttribute,
                                       {it->Arg(0), shadow_of(it->Arg(1))});
        }
    }
    motion.shadow = true;
    program.info.used_constant_buffer_types |= IR::Type::U32;
    return true;
}

} // namespace Shader::Optimization
