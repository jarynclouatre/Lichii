// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/spirv_inline_control.h"

namespace {
constexpr u32 Op(u32 opcode, u32 words) {
    return (words << 16) | opcode;
}

// A module skeleton: entry point %1 (main) calls %2 `calls_big` times and %3 `calls_small`
// times; %2 has `big_body` OpNop-like instructions, %3 has one. Returns the words and the
// indices of each function's control word.
struct Module {
    std::vector<u32> words;
    std::size_t main_control{};
    std::size_t big_control{};
    std::size_t small_control{};
};

Module Build(u32 calls_big, u32 calls_small, u32 big_body, u32 big_control = 0) {
    Module m;
    auto& w = m.words;
    w = {0x07230203, 0x00010600, 0, 100, 0};
    w.insert(w.end(), {Op(15, 4), 5, 1, 0x6e69616d}); // OpEntryPoint GLCompute %1 "main"
    const auto function = [&](u32 id, u32 control) {
        w.insert(w.end(), {Op(54, 5), 10, id, control, 11});
        return w.size() - 2;
    };
    const auto body = [&](u32 count) {
        w.insert(w.end(), {Op(248, 2), 50}); // OpLabel
        for (u32 i = 0; i < count; ++i) {
            w.insert(w.end(), {Op(1, 3), 10, 60 + i}); // OpUndef
        }
    };
    m.main_control = function(1, 0);
    body(0);
    for (u32 i = 0; i < calls_big; ++i) {
        w.insert(w.end(), {Op(57, 4), 10, 80 + i, 2});
    }
    for (u32 i = 0; i < calls_small; ++i) {
        w.insert(w.end(), {Op(57, 4), 10, 90 + i, 3});
    }
    w.insert(w.end(), {Op(253, 1), Op(56, 1)}); // OpReturn, OpFunctionEnd
    m.big_control = function(2, big_control);
    body(big_body);
    w.insert(w.end(), {Op(253, 1), Op(56, 1)});
    m.small_control = function(3, 0);
    body(1);
    w.insert(w.end(), {Op(253, 1), Op(56, 1)});
    return m;
}
} // Anonymous namespace

TEST_CASE("Large functions called from several places are marked DontInline",
          "[spirv_inline_control]") {
    Module m = Build(3, 5, 200);
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(m.words) == 1);
    REQUIRE(m.words[m.big_control] == 0x2);
    REQUIRE(m.words[m.small_control] == 0); // too small: inlining it costs nothing
    REQUIRE(m.words[m.main_control] == 0);  // the entry point is never marked
}

TEST_CASE("A function called once, or a forced Inline one, is handled", "[spirv_inline_control]") {
    Module once = Build(1, 0, 200);
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(once.words) == 0);
    REQUIRE(once.words[once.big_control] == 0);

    Module forced = Build(2, 0, 200, 0x1); // Inline
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(forced.words) == 1);
    REQUIRE(forced.words[forced.big_control] == 0x2); // Inline cleared, DontInline set
}

TEST_CASE("Thresholds are respected", "[spirv_inline_control]") {
    Module m = Build(2, 0, 50);
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(m.words, 128, 2) == 0);
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(m.words, 32, 2) == 1);
    Module n = Build(2, 0, 200);
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(n.words, 128, 3) == 0);
}

TEST_CASE("Malformed modules are left alone", "[spirv_inline_control]") {
    std::vector<u32> empty;
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(empty) == 0);

    Module bad_magic = Build(3, 0, 200);
    bad_magic.words[0] = 0;
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(bad_magic.words) == 0);
    REQUIRE(bad_magic.words[bad_magic.big_control] == 0);

    Module truncated = Build(3, 0, 200);
    truncated.words.pop_back();
    truncated.words.back() = Op(1, 9); // the last instruction claims words past the end
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(truncated.words) == 0);
    REQUIRE(truncated.words[truncated.big_control] == 0);

    Module zero_count = Build(3, 0, 200);
    zero_count.words[5] = 0; // an instruction with a word count of 0
    REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(zero_count.words) == 0);
}

TEST_CASE("Graphics stages are marked like compute", "[spirv_inline_control]") {
    // Same module with a Vertex (0) or Fragment (4) entry point instead of GLCompute.
    for (const u32 execution_model : {0u, 4u}) {
        Module m = Build(3, 5, 200);
        m.words[6] = execution_model;
        REQUIRE(VideoCommon::MarkSharedFunctionsDontInline(m.words) == 1);
        REQUIRE(m.words[m.big_control] == 0x2);
        REQUIRE(m.words[m.small_control] == 0);
        REQUIRE(m.words[m.main_control] == 0);
    }
}
