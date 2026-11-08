// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/native_qmd.h"

using Tegra::NativeQmdScheduler;
using Address = NativeQmdScheduler::Address;
using Descriptor = NativeQmdScheduler::Descriptor;
using Word = NativeQmdScheduler::Word;

namespace {

constexpr Word Enable = 1U << 23;
constexpr Word OneWord = 1U << 30;
constexpr Word Reduction = 1U << 26;

struct Released {
    Address address;
    Word payload;
    Word flags;
    bool operator==(const Released&) const = default;
};

struct Launch {
    std::string_view error;
    std::vector<Released> released;
};

Descriptor MakeQmd() {
    Descriptor qmd{};
    qmd[18] = 0x30;
    return qmd;
}

void SetRelease(Descriptor& qmd, Word base, Word address, Word flags, Word payload) {
    qmd[base] = address;
    qmd[base + 1] = flags;
    qmd[base + 2] = payload;
}

Launch Schedule(NativeQmdScheduler& scheduler, Address address, const std::map<Address, Descriptor>& memory) {
    Launch launch;
    launch.error = scheduler.Signal(
        address, 3,
        [&](Address qmd_address, Descriptor& descriptor) {
            const auto it = memory.find(qmd_address);
            if (it == memory.end()) return false;
            descriptor = it->second;
            return true;
        },
        [&](Address destination, Word payload, Word flags) {
            launch.released.push_back({destination, payload, flags});
            return true;
        });
    return launch;
}

} // Anonymous namespace

TEST_CASE("An undecodable QMD release slot keeps the other releases and the dependent QMD",
          "[video_core][native_qmd]") {
    // Orbitals (#400): both channels wait on 0x1b18871010, released by slot 0 of this QMD.
    constexpr Address first = 0x1b19635400;
    constexpr Address second = 0x1b19635500;
    auto qmd = MakeQmd();
    SetRelease(qmd, 24, 0x18871010, 0x4088001b, 6);
    SetRelease(qmd, 52, 0x18871020, Enable | OneWord | (1U << 29) | 0x1b, 7);
    qmd[15] = Word(second >> 8);
    qmd[16] = 1 | (1U << 1);
    auto dependent = MakeQmd();
    SetRelease(dependent, 24, 0x18871030, Enable | OneWord | 0x1b, 8);

    NativeQmdScheduler scheduler;
    const auto launch = Schedule(scheduler, first, {{first, qmd}, {second, dependent}});
    REQUIRE(launch.error == "unsupported QMD release format");
    REQUIRE(launch.released == std::vector<Released>{
                                   {0x1b18871010, 6, 0x4088001b},
                                   {0x1b18871030, 8, Enable | OneWord | 0x1b},
                               });
}

TEST_CASE("Timestamped, reduction, signed and interrupting QMD releases are performed",
          "[video_core][native_qmd]") {
    constexpr Address address = 0x20000;
    constexpr Word four_words = Enable;
    constexpr Word increment = Enable | OneWord | Reduction | (3U << 20);
    constexpr Word signed_max_trap = Enable | OneWord | Reduction | (1U << 24) | (2U << 20) | (1U << 27);
    auto qmd = MakeQmd();
    SetRelease(qmd, 24, 0x1000, four_words, 1);
    SetRelease(qmd, 28, 0x1010, increment, 2);
    SetRelease(qmd, 52, 0x1020, signed_max_trap, 3);

    NativeQmdScheduler scheduler;
    const auto launch = Schedule(scheduler, address, {{address, qmd}});
    REQUIRE(launch.error.empty());
    REQUIRE(launch.released == std::vector<Released>{
                                   {0x1000, 1, four_words},
                                   {0x1010, 2, increment},
                                   {0x1020, 3, signed_max_trap},
                               });
}

TEST_CASE("Skipped QMD release slots are reported with their raw words", "[video_core][native_qmd]") {
    constexpr Address address = 0x30000;
    constexpr Word two_words = Enable | (2U << 30);
    constexpr Word reserved_size = Enable | (3U << 30);
    constexpr Word timestamped_reduction = Enable | Reduction;
    auto qmd = MakeQmd();
    SetRelease(qmd, 24, 0x2000, two_words, 1);
    SetRelease(qmd, 28, 0x2010, reserved_size, 2);
    SetRelease(qmd, 52, 0x2020, timestamped_reduction, 3);

    NativeQmdScheduler scheduler;
    std::vector<std::array<Word, 6>> skipped;
    scheduler.skipped_release = [&](Address qmd_address, std::size_t slot, std::span<const Word, 4> words) {
        skipped.push_back({Word(qmd_address), Word(slot), words[0], words[1], words[2], words[3]});
    };
    const auto launch = Schedule(scheduler, address, {{address, qmd}});
    REQUIRE(launch.error == "unsupported QMD release format");
    REQUIRE(launch.released.empty());
    REQUIRE(skipped == std::vector<std::array<Word, 6>>{
                           {Word(address), 0, 0x2000, two_words, 1, 0},
                           {Word(address), 1, 0x2010, reserved_size, 2, 0},
                           {Word(address), 2, 0x2020, timestamped_reduction, 3, 0},
                       });
}

TEST_CASE("Semaphore reductions follow the documented ops", "[video_core][native_qmd]") {
    using Tegra::ReduceSemaphore;
    REQUIRE(ReduceSemaphore(0, false, 5, 3) == 8);
    REQUIRE(ReduceSemaphore(1, false, 0xffffffff, 1) == 1);
    REQUIRE(ReduceSemaphore(1, true, 0xffffffff, 1) == 0xffffffff);
    REQUIRE(ReduceSemaphore(2, false, 0xffffffff, 1) == 0xffffffff);
    REQUIRE(ReduceSemaphore(2, true, 0xffffffff, 1) == 1);
    REQUIRE(ReduceSemaphore(3, false, 4, 5) == 5);
    REQUIRE(ReduceSemaphore(3, false, 5, 5) == 0);
    REQUIRE(ReduceSemaphore(4, false, 3, 5) == 2);
    REQUIRE(ReduceSemaphore(4, false, 0, 5) == 5);
    REQUIRE(ReduceSemaphore(4, false, 7, 5) == 5);
    REQUIRE(ReduceSemaphore(5, false, 0b1100, 0b1010) == 0b1000);
    REQUIRE(ReduceSemaphore(6, false, 0b1100, 0b1010) == 0b1110);
    REQUIRE(ReduceSemaphore(7, false, 0b1100, 0b1010) == 0b0110);
}

TEST_CASE("A QMD scheduler with a table of its own schedules apart from the shared one",
          "[video_core][native_qmd]") {
    constexpr Address address = 0x40000;
    auto qmd = MakeQmd();
    qmd[4] = 1U << 14; // One dependency
    SetRelease(qmd, 24, 0x5000, Enable | OneWord, 1);
    const std::map<Address, Descriptor> memory{{address, qmd}};
    const auto signal = [&](NativeQmdScheduler& scheduler, Word action) {
        std::vector<Released> released;
        const auto error = scheduler.Signal(
            address, action,
            [&](Address qmd_address, Descriptor& descriptor) {
                const auto it = memory.find(qmd_address);
                if (it == memory.end()) {
                    return false;
                }
                descriptor = it->second;
                return true;
            },
            [&](Address destination, Word payload, Word flags) {
                released.push_back({destination, payload, flags});
                return true;
            });
        REQUIRE(error.empty());
        return released;
    };
    const std::vector<Released> run{{0x5000, 1, Enable | OneWord}};
    // Loaded on the shared table, it waits for its dependency.
    NativeQmdScheduler shared;
    REQUIRE(signal(shared, 3).empty());
    // A copy of the table runs it at the decrement as the shared one does, and apart from it.
    NativeQmdScheduler own{NativeQmdScheduler::OwnTable{}};
    own.CopyStates(shared);
    REQUIRE(signal(own, 7) == run);
    REQUIRE(signal(own, 7).empty()); // Completed: a decrement is kept for its next load
    REQUIRE(signal(shared, 7) == run);
    // A table that never saw the load keeps the decrement for it.
    NativeQmdScheduler other{NativeQmdScheduler::OwnTable{}};
    REQUIRE(signal(other, 7).empty());
    REQUIRE(signal(other, 3) == run);
}

TEST_CASE("A QMD scheduler takes back what an attempt changed and keeps what one committed",
          "[video_core][native_qmd]") {
    constexpr Address first = 0x40000;
    constexpr Address second = 0x40100;
    auto waiting = MakeQmd();
    waiting[4] = 1U << 14; // One dependency
    SetRelease(waiting, 24, 0x5000, Enable | OneWord, 1);
    auto chained = MakeQmd(); // Decrements `first` when it runs
    SetRelease(chained, 24, 0x5004, Enable | OneWord, 2);
    chained[15] = static_cast<Word>(first >> 8);
    chained[16] = 1 | (4U << 1);
    std::map<Address, Descriptor> memory{{first, waiting}, {second, chained}};
    bool unreadable{};
    const auto signal = [&](NativeQmdScheduler& scheduler, Address address, Word action) {
        std::vector<Released> released;
        const auto error = scheduler.Signal(
            address, action,
            [&](Address qmd_address, Descriptor& descriptor) {
                const auto it = memory.find(qmd_address);
                if (it == memory.end() || (unreadable && qmd_address == first)) {
                    return false;
                }
                descriptor = it->second;
                return true;
            },
            [&](Address destination, Word payload, Word flags) {
                released.push_back({destination, payload, flags});
                return true;
            });
        return std::make_pair(std::string{error}, released);
    };
    NativeQmdScheduler own{NativeQmdScheduler::OwnTable{}};
    NativeQmdScheduler reference{NativeQmdScheduler::OwnTable{}};
    REQUIRE(signal(own, first, 3).second.empty()); // Loaded, waits for its dependency
    REQUIRE(signal(reference, first, 3).second.empty());
    // The chained launch runs and decrements `first`, which can't be read: taken back, the table
    // is as before, `second` never ran.
    unreadable = true;
    own.BeginAttempt();
    REQUIRE_FALSE(signal(own, second, 3).first.empty());
    own.RollbackAttempt();
    REQUIRE(own.SameStates(reference));
    REQUIRE(own.StateCount() == 1);
    // Tried again once it can be read: both run, as on a table that never tried.
    unreadable = false;
    own.BeginAttempt();
    const auto ran = signal(own, second, 3);
    own.CommitAttempt();
    const std::vector<Released> both{{0x5004, 2, Enable | OneWord}, {0x5000, 1, Enable | OneWord}};
    REQUIRE(ran.first.empty());
    REQUIRE(ran.second == both);
    REQUIRE(signal(reference, second, 3).second == both);
    REQUIRE(own.SameStates(reference));
    // A rollback after the commit changes nothing.
    own.RollbackAttempt();
    REQUIRE(own.SameStates(reference));
}

TEST_CASE("A QMD scheduler takes back the history an attempt dropped at its limit",
          "[video_core][native_qmd]") {
    // Any address holds a QMD that runs at once; one of them can't be read.
    const Descriptor qmd = MakeQmd();
    Address unreadable = 0;
    const auto signal = [&](NativeQmdScheduler& scheduler, Address address) {
        return std::string{scheduler.Signal(
            address, 3,
            [&](Address qmd_address, Descriptor& descriptor) {
                descriptor = qmd;
                return qmd_address != unreadable;
            },
            [](Address, Word, Word) { return true; })};
    };
    NativeQmdScheduler own{NativeQmdScheduler::OwnTable{}};
    NativeQmdScheduler reference{NativeQmdScheduler::OwnTable{}};
    for (Address i = 0; i < 4096; ++i) {
        REQUIRE(signal(own, 0x100000 + i * 0x100).empty());
        REQUIRE(signal(reference, 0x100000 + i * 0x100).empty());
    }
    REQUIRE(own.StateCount() == 4096);
    // A new QMD past the limit drops the finished ones first; it can't be read: all taken back.
    unreadable = 0x900000;
    own.BeginAttempt();
    REQUIRE_FALSE(signal(own, unreadable).empty());
    own.RollbackAttempt();
    REQUIRE(own.StateCount() == 4096);
    REQUIRE(own.SameStates(reference));
    // Kept, it drops them as the reference does.
    unreadable = 0;
    own.BeginAttempt();
    REQUIRE(signal(own, 0x900000).empty());
    own.CommitAttempt();
    REQUIRE(signal(reference, 0x900000).empty());
    REQUIRE(own.StateCount() == 1);
    REQUIRE(own.SameStates(reference));
}
