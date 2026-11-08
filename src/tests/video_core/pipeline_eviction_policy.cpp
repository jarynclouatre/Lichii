// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/pipeline_eviction_policy.h"

namespace {

void CheckOldest(std::span<const u64> last_uses, size_t count) {
    std::vector<u64> candidates(last_uses.begin(), last_uses.end());
    Vulkan::OldestPipelineSelection selection{candidates, count};
    std::vector<u64> evicted;
    for (const u64 last_use : last_uses) {
        if (evicted.size() == count) {
            break;
        }
        if (selection.Take(last_use)) {
            evicted.push_back(last_use);
        }
    }
    std::vector<u64> expected(last_uses.begin(), last_uses.end());
    std::ranges::sort(expected);
    expected.resize(std::min(count, expected.size()));
    std::ranges::sort(evicted);
    CHECK(evicted == expected);
}

using Vulkan::EvictionCandidate;

std::vector<size_t> CheckSelection(std::span<const EvictionCandidate> candidates,
                                   size_t max_protected, size_t count) {
    std::vector<bool> is_protected(candidates.size());
    std::vector<u64> unprotected_last_uses;
    size_t protected_count = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!candidates[i].used && protected_count < max_protected) {
            is_protected[i] = true;
            ++protected_count;
        } else {
            unprotected_last_uses.push_back(candidates[i].last_use);
        }
    }
    const std::vector<size_t> selected =
        Vulkan::SelectPipelinesToEvict(candidates, max_protected, count);
    CHECK(std::ranges::is_sorted(selected));
    CHECK(std::ranges::adjacent_find(selected) == selected.end());
    std::vector<u64> selected_last_uses;
    for (const size_t i : selected) {
        REQUIRE(i < candidates.size());
        CHECK_FALSE(is_protected[i]);
        selected_last_uses.push_back(candidates[i].last_use);
    }
    std::ranges::sort(unprotected_last_uses);
    unprotected_last_uses.resize(std::min(count, unprotected_last_uses.size()));
    std::ranges::sort(selected_last_uses);
    CHECK(selected_last_uses == unprotected_last_uses);
    return selected;
}

} // namespace

TEST_CASE("Pipeline eviction preserves strict age ordering across cutoff ties",
          "[video_core][pipeline_eviction]") {
    std::array<u64, 3> last_uses{100, 200, 200};
    do {
        CAPTURE(last_uses);
        CheckOldest(last_uses, 2);
    } while (std::ranges::next_permutation(last_uses).found);
}

TEST_CASE("Pipeline eviction selects the requested number of oldest candidates",
          "[video_core][pipeline_eviction]") {
    const std::array<u64, 0> empty{};
    CheckOldest(empty, 0);
    CheckOldest(empty, 1);
    const std::array<u64, 4> mixed{300, 100, 200, 100};
    const std::array<u64, 4> equal{0, 0, 0, 0};
    for (size_t count = 0; count <= mixed.size() + 1; ++count) {
        CAPTURE(count);
        CheckOldest(mixed, count);
        CheckOldest(equal, count);
    }
}

TEST_CASE("Pipeline eviction handles the largest last-use frame without overflow",
          "[video_core][pipeline_eviction]") {
    constexpr u64 maximum = std::numeric_limits<u64>::max();
    const std::array<u64, 3> last_uses{maximum, maximum, maximum - 1};
    for (size_t count = 1; count <= last_uses.size(); ++count) {
        CAPTURE(count);
        CheckOldest(last_uses, count);
    }
}

TEST_CASE("Pipeline eviction skips protected never-used pipelines before choosing the oldest",
          "[video_core][pipeline_eviction]") {
    // Over budget by 500 with 1,500 protected prebuilt pipelines (last use 0) and 2,000 old used
    // ones: the 500 oldest used ones go, no protected one does.
    std::vector<EvictionCandidate> candidates;
    for (u64 i = 0; i < 3500; ++i) {
        if (i % 7 < 3) {
            candidates.push_back({0, false});
        } else {
            candidates.push_back({1 + (i * 7919) % 5000, true});
        }
    }
    REQUIRE(std::ranges::count_if(candidates, [](const EvictionCandidate& candidate) {
                return !candidate.used;
            }) == 1500);
    const std::vector<size_t> selected = CheckSelection(candidates, 2000, 500);
    CHECK(selected.size() == 500);
    for (const size_t i : selected) {
        CHECK(candidates[i].used);
    }
}

TEST_CASE("Pipeline eviction evicts never-used pipelines past the protected limit first",
          "[video_core][pipeline_eviction]") {
    // 5 never-used, 2 protected: the 3 unprotected never-used ones (last use 0) are the oldest.
    const std::array<EvictionCandidate, 8> candidates{{
        {0, false}, {40, true}, {0, false}, {0, false}, {10, true}, {0, false}, {0, false},
        {20, true}}};
    CHECK(CheckSelection(candidates, 2, 3) == std::vector<size_t>{3, 5, 6});
    CHECK(CheckSelection(candidates, 2, 4) == std::vector<size_t>{3, 4, 5, 6});
    CHECK(CheckSelection(candidates, 2, 100) == std::vector<size_t>{1, 3, 4, 5, 6, 7});
    CHECK(CheckSelection(candidates, 2, 0).empty());
    // Nothing protected, as on a title whose prebuild is past the limit.
    CHECK(CheckSelection(candidates, 0, 5) == std::vector<size_t>{0, 2, 3, 5, 6});
    // No count limit, as on a Switch 1 title: every unprotected one goes.
    CHECK(CheckSelection(candidates, 2, std::numeric_limits<size_t>::max()) ==
          std::vector<size_t>{1, 3, 4, 5, 6, 7});
    // Only protected ones are old enough: the budget is exceeded instead.
    const std::array<EvictionCandidate, 2> only_prebuilt{{{0, false}, {0, false}}};
    CHECK(CheckSelection(only_prebuilt, 2000, 2).empty());
}

TEST_CASE("Pipeline eviction keeps an exact count across ties at frame 0",
          "[video_core][pipeline_eviction]") {
    // A pipeline drawn on frame 0 has last use 0 like a prebuilt one but is not protected.
    std::array<EvictionCandidate, 6> candidates{{
        {0, true}, {0, false}, {0, true}, {0, false}, {5, true}, {0, false}}};
    for (size_t max_protected = 0; max_protected <= 4; ++max_protected) {
        for (size_t count = 0; count <= candidates.size() + 1; ++count) {
            CAPTURE(max_protected, count);
            CheckSelection(candidates, max_protected, count);
        }
    }
}

TEST_CASE("Pipeline eviction matches a reference selection on random caches",
          "[video_core][pipeline_eviction]") {
    std::mt19937_64 rng{313};
    for (int round = 0; round < 2000; ++round) {
        const size_t size = rng() % 64;
        const u64 ages = 1 + rng() % 8;
        std::vector<EvictionCandidate> candidates;
        for (size_t i = 0; i < size; ++i) {
            const bool used = rng() % 3 != 0;
            candidates.push_back({used ? rng() % ages : 0, used});
        }
        const size_t max_protected = rng() % 16;
        const size_t count = rng() % (size + 2);
        CAPTURE(round, max_protected, count);
        CheckSelection(candidates, max_protected, count);
    }
}
