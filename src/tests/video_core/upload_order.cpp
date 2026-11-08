// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <random>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/upload_order.h"

namespace {

struct Copy {
    u64 dst_offset;
    u64 size;
};

/// The tick of each byte's last use, and every tick that used any byte.
struct ByteModel {
    explicit ByteModel(size_t size) : last(size, 0) {}

    void Track(u64 offset, u64 size, u64 tick) {
        for (u64 i = offset; i < offset + size; ++i) {
            last[i] = tick;
        }
        if (size != 0) {
            ticks.push_back(tick);
        }
    }

    bool IsUsed(u64 offset, u64 size, u64 tick) const {
        for (u64 i = offset; i < offset + size; ++i) {
            if (last[i] == tick) {
                return true;
            }
        }
        return false;
    }

    /// The newest tick before `tick` that used a byte of [offset, offset + size).
    u64 NewestByteUseBefore(u64 offset, u64 size, u64 tick) const {
        u64 newest = 0;
        for (u64 i = offset; i < offset + size; ++i) {
            if (last[i] < tick) {
                newest = (std::max)(newest, last[i]);
            }
        }
        return newest;
    }

    u64 NewestTickBefore(u64 tick) const {
        u64 newest = 0;
        for (const u64 used : ticks) {
            if (used < tick) {
                newest = (std::max)(newest, used);
            }
        }
        return newest;
    }

    std::vector<u64> last;
    std::vector<u64> ticks;
};

} // Anonymous namespace

TEST_CASE("Submission uses keep the newest submission's byte ranges", "[video_core]") {
    VideoCommon::SubmissionUses uses;
    REQUIRE_FALSE(uses.IsUsed(0, 16, 1));
    REQUIRE(uses.NewestTickBefore(1) == 0);

    uses.Track(256, 64, 5);
    uses.Track(320, 64, 5); // touches the end of one: merged
    uses.Track(1024, 16, 5);
    uses.Track(1008, 16, 5); // touches the start of one: merged
    REQUIRE(uses.RangeCount() == 2);
    REQUIRE(uses.IsUsed(256, 1, 5));
    REQUIRE(uses.IsUsed(383, 1, 5));
    REQUIRE_FALSE(uses.IsUsed(384, 624, 5)); // the gap between the ranges
    REQUIRE(uses.IsUsed(384, 625, 5));
    REQUIRE_FALSE(uses.IsUsed(192, 64, 5)); // ends where a range begins
    REQUIRE(uses.IsUsed(192, 65, 5));
    REQUIRE(uses.IsUsed(1000, 100, 5));
    REQUIRE_FALSE(uses.IsUsed(992, 16, 5));
    REQUIRE_FALSE(uses.IsUsed(1040, 16, 5));
    REQUIRE_FALSE(uses.IsUsed(256, 64, 4)); // another submission's
    REQUIRE_FALSE(uses.IsUsed(256, 0, 5));
    REQUIRE(uses.NewestTickBefore(5) == 0);
    REQUIRE(uses.NewestTickBefore(6) == 5);

    // A range covering several merges them.
    uses.Track(300, 800, 5);
    REQUIRE(uses.RangeCount() == 1);
    REQUIRE(uses.IsUsed(1050, 4, 5));

    // The next submission starts empty; the one before is the newest earlier use.
    uses.Track(4096, 4, 7);
    REQUIRE(uses.RangeCount() == 1);
    REQUIRE_FALSE(uses.IsUsed(256, 64, 7));
    REQUIRE(uses.IsUsed(4096, 4, 7));
    REQUIRE(uses.NewestTickBefore(7) == 5);
    REQUIRE(uses.NewestTickBefore(8) == 7);
}

TEST_CASE("Submission uses match a byte model", "[video_core]") {
    constexpr u64 Size = 16384;
    std::mt19937 random{0x0de7};
    for (u32 run = 0; run < 24; ++run) {
        VideoCommon::SubmissionUses uses;
        ByteModel model{Size};
        u64 tick = 1;
        for (u32 step = 0; step < 3000; ++step) {
            const u32 roll = random() % 100;
            if (roll < 3) {
                tick += 1 + random() % 3;
            }
            const u64 size = random() % 8 == 0 ? random() % 4096 : 1 + random() % 96;
            const u64 offset = random() % (Size - size);
            if (roll < 60) {
                uses.Track(offset, size, tick);
                model.Track(offset, size, tick);
                continue;
            }
            // Ask about the submission being recorded, as an upload does.
            REQUIRE(uses.IsUsed(offset, size, tick) == model.IsUsed(offset, size, tick));
            const u64 bound = uses.NewestTickBefore(tick);
            REQUIRE(bound == model.NewestTickBefore(tick));
            REQUIRE(bound >= model.NewestByteUseBefore(offset, size, tick));
            const VideoCommon::UploadOrder order =
                VideoCommon::DecideUploadOrder(uses, std::array{Copy{offset, size}}, tick);
            REQUIRE(order.reorder == !model.IsUsed(offset, size, tick));
            if (order.reorder) {
                REQUIRE(order.prior_tick == bound);
            }
        }
    }
}

TEST_CASE("Submission uses only grow past their range limit", "[video_core]") {
    constexpr u64 Size = (VideoCommon::SubmissionUses::MaxRanges + 64) * 64;
    VideoCommon::SubmissionUses uses;
    ByteModel model{Size};
    std::mt19937 random{0x51de};
    // Disjoint 16-byte ranges in a random order, more of them than the limit.
    std::vector<u64> slots(Size / 64);
    for (u64 i = 0; i < slots.size(); ++i) {
        slots[i] = i * 64;
    }
    std::shuffle(slots.begin(), slots.end(), random);
    for (const u64 offset : slots) {
        uses.Track(offset, 16, 3);
        model.Track(offset, 16, 3);
        REQUIRE(uses.RangeCount() <= VideoCommon::SubmissionUses::MaxRanges);
    }
    // Every used byte still is; unused ones may be counted as used, never the reverse.
    for (u64 offset = 0; offset < Size; offset += 8) {
        if (model.IsUsed(offset, 8, 3)) {
            REQUIRE(uses.IsUsed(offset, 8, 3));
        }
    }
    REQUIRE_FALSE(uses.IsUsed(Size, 64, 3));
}

TEST_CASE("Uploads stay in order behind uses in their submission", "[video_core]") {
    VideoCommon::SubmissionUses uses;
    // Submission 4 drew from bytes 0..256.
    uses.Track(0, 256, 4);
    // Submission 5: the first upload goes ahead, after submission 4's work.
    const auto first = VideoCommon::DecideUploadOrder(uses, std::array{Copy{128, 64}}, 5);
    REQUIRE(first.reorder);
    REQUIRE(first.prior_tick == 4);
    uses.Track(128, 64, 5);
    // A draw then reads other bytes; an upload of those must stay behind it.
    uses.Track(512, 64, 5);
    REQUIRE_FALSE(VideoCommon::DecideUploadOrder(uses, std::array{Copy{540, 4}}, 5).reorder);
    // An upload of the first upload's bytes stays behind it too.
    REQUIRE_FALSE(VideoCommon::DecideUploadOrder(uses, std::array{Copy{160, 8}}, 5).reorder);
    // One copy of several is enough.
    REQUIRE_FALSE(VideoCommon::DecideUploadOrder(
                      uses, std::array{Copy{1024, 8}, Copy{560, 8}, Copy{2048, 8}}, 5)
                      .reorder);
    // Untouched bytes go ahead.
    const auto other = VideoCommon::DecideUploadOrder(uses, std::array{Copy{1024, 8}}, 5);
    REQUIRE(other.reorder);
    REQUIRE(other.prior_tick == 4);
    // A buffer no work used: ahead, after nothing.
    const VideoCommon::SubmissionUses unused;
    const auto fresh = VideoCommon::DecideUploadOrder(unused, std::array{Copy{0, 4}}, 5);
    REQUIRE(fresh.reorder);
    REQUIRE(fresh.prior_tick == 0);
}
