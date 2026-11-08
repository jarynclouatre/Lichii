// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "video_core/spirv_inline_control.h"

namespace VideoCommon {
namespace {
constexpr u32 SpirvMagic = 0x07230203;
constexpr u32 OpEntryPoint = 15;
constexpr u32 OpFunction = 54;
constexpr u32 OpFunctionEnd = 56;
constexpr u32 OpFunctionCall = 57;
constexpr u32 FunctionControlInline = 0x1;
constexpr u32 FunctionControlDontInline = 0x2;
constexpr std::size_t HeaderWords = 5;
} // Anonymous namespace

std::size_t MarkSharedFunctionsDontInline(std::span<u32> words, std::size_t min_instructions,
                                          std::size_t min_calls) {
    if (words.size() < HeaderWords || words[0] != SpirvMagic) {
        return 0;
    }
    struct Function {
        std::size_t control_word{};
        std::size_t instructions{};
        std::size_t calls{};
    };
    std::unordered_set<u32> entry_points;
    std::unordered_map<u32, Function> functions;
    Function* current = nullptr;
    for (std::size_t index = HeaderWords; index < words.size();) {
        const u32 count = words[index] >> 16;
        const u32 opcode = words[index] & 0xffff;
        if (count == 0 || count > words.size() - index) {
            return 0;
        }
        switch (opcode) {
        case OpEntryPoint:
            if (count < 3) {
                return 0;
            }
            entry_points.insert(words[index + 2]);
            break;
        case OpFunction:
            // OpFunction <result type> <result id> <function control> <function type>
            if (count != 5 || current != nullptr) {
                return 0;
            }
            current = &functions[words[index + 2]];
            current->control_word = index + 3;
            break;
        case OpFunctionCall:
            if (count < 4) {
                return 0;
            }
            ++functions[words[index + 3]].calls;
            break;
        default:
            break;
        }
        if (current != nullptr) {
            ++current->instructions;
        }
        if (opcode == OpFunctionEnd) {
            current = nullptr;
        }
        index += count;
    }
    std::size_t marked = 0;
    for (const auto& [id, function] : functions) {
        if (function.control_word == 0 || entry_points.contains(id) ||
            function.instructions < min_instructions || function.calls < min_calls) {
            continue;
        }
        u32& control = words[function.control_word];
        control = (control & ~FunctionControlInline) | FunctionControlDontInline;
        ++marked;
    }
    return marked;
}

} // namespace VideoCommon
