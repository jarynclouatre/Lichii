// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <initializer_list>
#include <random>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/macro_turing.h"

namespace {

using Effects = Tegra::TuringMacroInterpreter::Effects;

/// A Turing MME instruction from its fields (TuringMacroInterpreter::Decode); a lane left alone
/// adds the zero register to itself.
struct Mme {
    bool end{};
    u32 pred_mode{};
    u32 pred_reg{};
    std::array<u32, 2> op{}, dst{24, 24}, src_a{24, 24}, src_b{24, 24}, imm{};
    std::array<u32, 2> out_method{}, out_emit{};
};

std::vector<u32> Code(std::span<const Mme> instructions) {
    std::vector<u32> code;
    for (const Mme& f : instructions) {
        std::array<u32, 3> bits{};
        const auto field = [&bits](u32 offset, u32 count, u32 value) {
            for (u32 bit = 0; bit < count; ++bit) {
                bits[(offset + bit) / 32] |= ((value >> bit) & 1) << ((offset + bit) % 32);
            }
        };
        field(0, 1, f.end ? 1 : 0);
        field(1, 4, f.pred_mode);
        field(5, 5, f.pred_reg);
        for (u32 lane = 0; lane < 2; ++lane) {
            field(10 + lane * 36, 5, f.op[lane]);
            field(15 + lane * 36, 5, f.dst[lane]);
            field(20 + lane * 36, 5, f.src_a[lane]);
            field(25 + lane * 36, 5, f.src_b[lane]);
            field(30 + lane * 36, 16, f.imm[lane]);
            field(82 + lane * 7, 3, f.out_method[lane]);
            field(85 + lane * 7, 4, f.out_emit[lane]);
        }
        code.insert(code.end(), {bits[2], bits[1], bits[0]});
    }
    return code;
}

/// Sends the next parameter to `method` (an immediate with its increment in bits 12-15).
Mme Fifo(u32 method, bool end = false) {
    return Mme{.end = end, .imm = {method, 0}, .out_method = {5, 0}, .out_emit = {3, 0}};
}

Effects Analyze(std::initializer_list<Mme> instructions, u32 start = 0) {
    return Tegra::TuringMacroInterpreter::AnalyzeEffects(
        Code(std::span<const Mme>{instructions.begin(), instructions.size()}), start);
}

} // Anonymous namespace

TEST_CASE("Turing macro effects: a FIFO read writes its registers, up to the one after END",
          "[video_core][macro]") {
    // Ocarina of Time's render enable macro reads six words at the address in its parameters.
    const auto effects = Analyze({Fifo(0x154), Fifo(0x155), Fifo(0x158, true), Fifo(0x556),
                                  Fifo(0x557)});
    CHECK(effects.methods_known);
    CHECK_FALSE(effects.data_ram);
    CHECK(effects.methods == std::vector<u32>{0x154, 0x155, 0x158, 0x556});
    // The same code entered past its END: the call runs to the end of the code.
    CHECK(Analyze({Fifo(0x154), Fifo(0x155), Fifo(0x158, true), Fifo(0x556), Fifo(0x557)}, 3)
              .methods == std::vector<u32>{0x556, 0x557});
}

TEST_CASE("Turing macro effects: increments, branches and predicates", "[video_core][macro]") {
    // One method with an increment of 1, written twice: both registers.
    const Mme twice{.end = true,
                    .imm = {0x1100, 0},
                    .out_method = {5, 0},
                    .out_emit = {3, 3}};
    CHECK(Analyze({twice, Fifo(0x200)}).methods == std::vector<u32>{0x100, 0x101, 0x200});
    // r0 = parameter; a branch on r0 == 0 skips the write of 0x300: both ways count.
    const Mme load{.dst = {0, 24}, .src_a = {28, 24}};
    const Mme skip{.op = {29, 0}, .src_a = {0, 24}, .imm = {0x8000 | 2, 0}};
    const auto branched = Analyze({load, skip, Fifo(0x300), Fifo(0x301, true), Fifo(0x302)});
    CHECK(branched.methods == std::vector<u32>{0x300, 0x301, 0x302});
    // A predicate on the zero register: FFFF runs nothing of the instruction (its END stays).
    Mme predicated = Fifo(0x400, true);
    predicated.pred_mode = 2;
    predicated.pred_reg = 24;
    CHECK(Analyze({predicated, Fifo(0x401), Fifo(0x402)}).methods == std::vector<u32>{0x401});
}

TEST_CASE("Turing macro effects: what keeps a macro's writes unknown", "[video_core][macro]") {
    // A method taken from a parameter.
    const Mme from_fifo{.end = true, .out_method = {3, 0}, .out_emit = {4, 0}};
    CHECK_FALSE(Analyze({from_fifo, Fifo(0x100)}).methods_known);
    // A loop over the parameters writing consecutive registers (increment 1).
    const Mme count{.dst = {0, 24}, .src_a = {28, 24}};
    const Mme method{.imm = {0x1100, 0}, .out_method = {5, 0}};
    const Mme loop{.op = {23, 0}, .src_a = {0, 24}, .imm = {2, 0}};
    const Mme body{.out_emit = {3, 0}};
    CHECK_FALSE(Analyze({count, method, loop, body, Fifo(0x200, true), Fifo(0x201)}).methods_known);
    // The same loop without the increment writes one register.
    const Mme same{.imm = {0x100, 0}, .out_method = {5, 0}};
    const auto looped = Analyze({count, same, loop, body, Fifo(0x200, true), Fifo(0x201)});
    CHECK(looped.methods_known);
    CHECK(looped.methods == std::vector<u32>{0x100, 0x200, 0x201});
    // A data RAM write (DWRITE) in the delay slot after END.
    const Mme dwrite{.op = {31, 0}};
    CHECK(Analyze({Fifo(0x158, true), dwrite}).data_ram);
    CHECK_FALSE(Analyze({Fifo(0x158, true), Fifo(0x100), dwrite}).data_ram);
}

TEST_CASE("Turing macro effects cover every write Execute makes on random code",
          "[video_core][macro]") {
    std::mt19937 rng{7};
    const auto pick = [&rng](u32 count) { return static_cast<u32>(rng() % count); };
    for (u32 program = 0; program < 300; ++program) {
        // Registers 0-3 set first, then ALU, register read, flow and data RAM instructions with
        // methods from immediates mostly.
        std::vector<Mme> instructions(2);
        for (u32 i = 0; i < 4; ++i) {
            instructions[i / 2].dst[i % 2] = i;
            instructions[i / 2].src_a[i % 2] = pick(2) != 0 ? 25 : 28 + i % 2;
            instructions[i / 2].imm[i % 2] = pick(4);
        }
        const u32 length = 2 + pick(20);
        for (u32 pc = 0; pc < length; ++pc) {
            Mme f{.end = pick(10) == 0};
            if (pick(6) == 0) {
                f.pred_mode = pick(16);
                f.pred_reg = pick(5) == 0 ? 24 : pick(4);
            }
            for (u32 lane = 0; lane < 2; ++lane) {
                constexpr std::array<u32, 12> ops{0, 2, 9, 12, 14, 16, 17, 22, 23, 24, 25, 31};
                f.op[lane] = ops[pick(static_cast<u32>(ops.size()))];
                f.dst[lane] = pick(3) == 0 ? 24 : pick(4);
                f.src_a[lane] = pick(2) != 0 ? pick(4) : 24 + pick(6);
                f.src_b[lane] = pick(2) != 0 ? pick(4) : 24 + pick(6);
                if (f.op[lane] == 23) {
                    f.imm[lane] = 2 + pick(4);
                } else if (f.op[lane] == 24 || f.op[lane] == 25) {
                    // Short jumps and branches either way, some stopping the call.
                    const int offset = pick(8) == 0 ? -4096 : static_cast<int>(pick(9)) - 4;
                    f.imm[lane] = (static_cast<u32>(offset) & 0x1fff) | (pick(2) << 15) |
                                  (pick(2) << 14);
                } else {
                    f.imm[lane] = pick(3) == 0 ? 0x100 + pick(8) + (pick(3) << 12) : pick(0x10000);
                }
                f.out_method[lane] = pick(3) == 0 ? 0 : (pick(4) == 0 ? 1 + pick(4) : 5 + lane);
                f.out_emit[lane] = pick(3) == 0 ? 0 : (pick(3) == 0 ? 1 + pick(2) : 3 + lane);
            }
            instructions.push_back(f);
        }
        const auto code = Code(instructions);
        const u32 start = pick(4) == 0 ? pick(length + 2) : 0;
        const auto effects = Tegra::TuringMacroInterpreter::AnalyzeEffects(code, start);
        for (u32 run = 0; run < 4; ++run) {
            Tegra::TuringMacroInterpreter interpreter;
            std::vector<u32> fifo(4 + pick(28));
            std::ranges::generate(fifo, [&] { return pick(16) == 0 ? rng() : pick(8); });
            std::vector<u32> written;
            interpreter.Execute(
                code, start, fifo, [&](u32) { return pick(8); },
                [&](u32 method, u32) {
                    written.push_back(method);
                    return written.size() < 512;
                });
            CAPTURE(program, run);
            if (effects.methods_known) {
                for (const u32 method : written) {
                    CAPTURE(method);
                    CHECK(std::ranges::binary_search(effects.methods, method));
                }
            }
            if (!effects.data_ram) {
                CHECK(interpreter.data_ram_reads.empty());
                CHECK(interpreter.data_ram_writes.empty());
            }
        }
    }
}
