// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.h"

namespace Shader::Maxwell {

IR::FmzMode HalfPrecision2FmzMode(HalfPrecision precision) {
    switch (precision) {
    case HalfPrecision::None:
        return IR::FmzMode::None;
    case HalfPrecision::FTZ:
        return IR::FmzMode::FTZ;
    case HalfPrecision::FMZ:
        return IR::FmzMode::FMZ;
    default:
        return IR::FmzMode::DontCare;
    }
}

IR::F16F32F64 SelectFmzProduct(IR::IREmitter& ir, const IR::F16F32F64& a, const IR::F16F32F64& b,
                               const IR::F16F32F64& if_zero, const IR::F16F32F64& value) {
    const auto f32 = [&](const IR::F16F32F64& operand) {
        return IR::F32{ir.FPConvert(32, operand)};
    };
    const IR::F32 zero{ir.Imm32(0.0f)};
    const IR::U1 any_zero{ir.LogicalOr(ir.FPEqual(f32(a), zero), ir.FPEqual(f32(b), zero))};
    const IR::F32 result{ir.Select(any_zero, f32(if_zero), f32(value))};
    return ir.FPConvert(value.Type() == IR::Type::F16 ? 16 : 32, result);
}

std::pair<IR::F16F32F64, IR::F16F32F64> Extract(IR::IREmitter& ir, IR::U32 value, Swizzle swizzle) {
    switch (swizzle) {
    case Swizzle::H1_H0: {
        const IR::Value vector{ir.UnpackFloat2x16(value)};
        return {IR::F16{ir.CompositeExtract(vector, 0)}, IR::F16{ir.CompositeExtract(vector, 1)}};
    }
    case Swizzle::H0_H0: {
        const IR::F16 scalar{ir.CompositeExtract(ir.UnpackFloat2x16(value), 0)};
        return {scalar, scalar};
    }
    case Swizzle::H1_H1: {
        const IR::F16 scalar{ir.CompositeExtract(ir.UnpackFloat2x16(value), 1)};
        return {scalar, scalar};
    }
    case Swizzle::F32: {
        const IR::F32 scalar{ir.BitCast<IR::F32>(value)};
        return {scalar, scalar};
    }
    }
    throw InvalidArgument("Invalid swizzle {}", swizzle);
}

IR::U32 MergeResult(IR::IREmitter& ir, IR::Reg dest, const IR::F16& lhs, const IR::F16& rhs,
                    Merge merge) {
    switch (merge) {
    case Merge::H1_H0:
        return ir.PackFloat2x16(ir.CompositeConstruct(lhs, rhs));
    case Merge::F32:
        return ir.BitCast<IR::U32, IR::F32>(ir.FPConvert(32, lhs));
    case Merge::MRG_H0:
    case Merge::MRG_H1: {
        const IR::Value vector{ir.UnpackFloat2x16(ir.GetReg(dest))};
        const bool is_h0{merge == Merge::MRG_H0};
        const IR::F16 insert{ir.FPConvert(16, is_h0 ? lhs : rhs)};
        return ir.PackFloat2x16(ir.CompositeInsert(vector, insert, is_h0 ? 0 : 1));
    }
    }
    throw InvalidArgument("Invalid merge {}", merge);
}

} // namespace Shader::Maxwell
