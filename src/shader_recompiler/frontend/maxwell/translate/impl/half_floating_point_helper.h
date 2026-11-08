// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/common_types.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {

enum class Merge : u64 {
    H1_H0,
    F32,
    MRG_H0,
    MRG_H1,
};

enum class Swizzle : u64 {
    H1_H0,
    F32,
    H0_H0,
    H1_H1,
};

enum class HalfPrecision : u64 {
    None = 0,
    FTZ = 1,
    FMZ = 2,
};

IR::FmzMode HalfPrecision2FmzMode(HalfPrecision precision);

/// FMZ: a product with a zero factor is zero, even when the other factor is Inf or NaN. Returns
/// if_zero where a or b is zero and value elsewhere, with the type of value. F16 values are
/// compared and selected as F32, which is exact both ways: the IR has no F16 select.
IR::F16F32F64 SelectFmzProduct(IR::IREmitter& ir, const IR::F16F32F64& a, const IR::F16F32F64& b,
                               const IR::F16F32F64& if_zero, const IR::F16F32F64& value);

std::pair<IR::F16F32F64, IR::F16F32F64> Extract(IR::IREmitter& ir, IR::U32 value, Swizzle swizzle);

IR::U32 MergeResult(IR::IREmitter& ir, IR::Reg dest, const IR::F16& lhs, const IR::F16& rhs,
                    Merge merge);

} // namespace Shader::Maxwell
