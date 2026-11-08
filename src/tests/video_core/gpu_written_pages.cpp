// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2.h"

TEST_CASE("Nx2MayBeGpuWritten sees every page a GPU write marked", "[video_core][nx2]") {
    constexpr u64 page = u64{1} << Tegra::Nx2WrittenPageBits;
    constexpr u64 base = 0x10000000;
    Tegra::Nx2ClearGpuWrittenPages();
    CHECK_FALSE(Tegra::Nx2MayBeGpuWritten(base, 64 * page));

    Tegra::Nx2MarkGpuWritten(base + 3 * page + 100, 8);
    CHECK(Tegra::Nx2MayBeGpuWritten(base + 3 * page, 1));
    CHECK(Tegra::Nx2MayBeGpuWritten(base + 4 * page - 1, 1));
    CHECK(Tegra::Nx2MayBeGpuWritten(base, 3 * page + 1));
    CHECK(Tegra::Nx2MayBeGpuWritten(base + 3 * page - 4, 8));
    CHECK(Tegra::Nx2MayBeGpuWritten(base, 64 * page));
    CHECK_FALSE(Tegra::Nx2MayBeGpuWritten(base, 3 * page));
    CHECK_FALSE(Tegra::Nx2MayBeGpuWritten(base + 4 * page, 60 * page));
    CHECK_FALSE(Tegra::Nx2MayBeGpuWritten(base + 3 * page, 0));

    Tegra::Nx2MarkGpuWritten(base + 63 * page + page - 1, 2);
    CHECK(Tegra::Nx2MayBeGpuWritten(base + 64 * page, 1));
    CHECK_FALSE(Tegra::Nx2MayBeGpuWritten(base + 65 * page, page));

    const u64 bitmap_end = Tegra::g_nx2_gpu_written_pages.size() * 64 * page;
    CHECK(Tegra::Nx2MayBeGpuWritten(bitmap_end - 4, 8));
    CHECK(Tegra::Nx2MayBeGpuWritten(bitmap_end, page));
    CHECK(Tegra::Nx2MayBeGpuWritten(~u64{0} - 3, 8));

    Tegra::Nx2ClearGpuWrittenPages();
    CHECK_FALSE(Tegra::Nx2MayBeGpuWritten(base, 64 * page));
}

TEST_CASE("Nx2MarkGpuWritten marks exactly the pages a range touches", "[video_core][nx2]") {
    constexpr u64 page = u64{1} << Tegra::Nx2WrittenPageBits;
    constexpr u64 base = 0x20000000;
    constexpr u64 first_word = base / page / 64;
    std::mt19937_64 rng{7};
    for (int round = 0; round < 200; ++round) {
        Tegra::Nx2ClearGpuWrittenPages();
        std::array<u64, 8> expected{};
        for (int write = 0; write < 3; ++write) {
            const u64 offset = rng() % (6 * 64 * page);
            const u64 size = 1 + rng() % (2 * 64 * page);
            Tegra::Nx2MarkGpuWritten(base + offset, size);
            for (u64 p = offset / page; p <= (offset + size - 1) / page; ++p) {
                expected[p / 64] |= u64{1} << (p % 64);
            }
        }
        for (u64 word = 0; word < expected.size(); ++word) {
            REQUIRE(Tegra::g_nx2_gpu_written_pages[first_word + word].load() == expected[word]);
        }
        REQUIRE(Tegra::g_nx2_gpu_written_pages[first_word - 1].load() == 0);
    }
    const u64 bitmap_pages = Tegra::g_nx2_gpu_written_pages.size() * 64;
    Tegra::Nx2MarkGpuWritten((bitmap_pages - 3) * page, 16 * page);
    CHECK(Tegra::g_nx2_gpu_written_pages.back().load() == u64{7} << 61);
    Tegra::Nx2ClearGpuWrittenPages();
}
