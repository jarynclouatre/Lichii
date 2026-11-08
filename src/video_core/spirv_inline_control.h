// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>

#include "common/common_types.h"

namespace VideoCommon {

/// Marks DontInline (and clears Inline on) every function of a SPIR-V module that is not an entry
/// point, has at least `min_instructions` instructions and is called from at least `min_calls`
/// places. Returns how many functions it marked; a malformed module is left as it is (0).
///
/// The global memory functions of a native program (spirv_emit_context.cpp,
/// DefineGlobalMemoryFunctions) test the address against every storage buffer, several hundred
/// instructions each, and a compute shader calls them from dozens of places. NVIDIA's compiler
/// inlines every call: a test title's compute shaders took 2-13 s to build (RTX 3080 Ti,
/// 615.78) and froze the emulator at their first dispatch. Called, they build in a fraction of
/// that.
std::size_t MarkSharedFunctionsDontInline(std::span<u32> words, std::size_t min_instructions = 128,
                                          std::size_t min_calls = 2);

} // namespace VideoCommon
