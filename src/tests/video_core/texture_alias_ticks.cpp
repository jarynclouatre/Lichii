// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "common/slot_vector.h"
#include "video_core/texture_cache/image_base.h"

namespace {

using VideoCommon::AliasedImage;
using VideoCommon::ImageBase;
using VideoCommon::ImageId;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::SetModificationTick;
using VideoCore::Surface::PixelFormat;

ImageInfo Info() {
    ImageInfo info;
    info.format = PixelFormat::A8B8G8R8_UNORM;
    info.type = ImageType::e2D;
    info.size = {64, 64, 1};
    info.resources.levels = 1;
    info.resources.layers = 1;
    return info;
}

/// What TextureCache::SynchronizeAliases relies on to return early.
bool NewestAliasTickHolds(Common::SlotVector<ImageBase>& images, ImageId id) {
    const ImageBase& image = images[id];
    return std::ranges::all_of(image.aliased_images, [&](const AliasedImage& alias) {
        return images[alias.id].modification_tick <= image.newest_alias_tick;
    });
}

} // namespace

TEST_CASE("Aliases see the tick a joined image takes from its copies",
          "[video_core][texture_cache]") {
    Common::SlotVector<ImageBase> images;
    const ImageId newer = images.insert(Info(), 0x10000, 0x10000);
    const ImageId older = images.insert(Info(), 0x10000, 0x10000);
    const ImageId joined = images.insert(Info(), 0x10000, 0x10000);
    SetModificationTick(images, images[older], 3);
    SetModificationTick(images, images[newer], 5);

    // As TextureCache::JoinImages: aliases are added first, then the copies set the new image's
    // tick from its overlaps, oldest first.
    REQUIRE(AddImageAlias(images[joined], images[older], joined, older));
    REQUIRE(AddImageAlias(images[joined], images[newer], joined, newer));
    SetModificationTick(images, images[joined], images[older].modification_tick);
    SetModificationTick(images, images[joined], images[newer].modification_tick);

    for (const ImageId id : {newer, older, joined}) {
        CHECK(NewestAliasTickHolds(images, id));
    }
    // The older alias must not skip its copy from the joined image.
    CHECK(images[older].newest_alias_tick > images[older].modification_tick);
    CHECK(images[newer].newest_alias_tick <= images[newer].modification_tick);
}

TEST_CASE("Adding an alias raises both images' newest alias tick",
          "[video_core][texture_cache]") {
    Common::SlotVector<ImageBase> images;
    const ImageId a = images.insert(Info(), 0x10000, 0x10000);
    const ImageId b = images.insert(Info(), 0x10000, 0x10000);
    SetModificationTick(images, images[a], 7);
    SetModificationTick(images, images[b], 2);
    REQUIRE(AddImageAlias(images[a], images[b], a, b));
    CHECK(images[a].newest_alias_tick == 2);
    CHECK(images[b].newest_alias_tick == 7);
    SetModificationTick(images, images[b], 9);
    CHECK(images[a].newest_alias_tick == 9);
    CHECK(NewestAliasTickHolds(images, a));
    CHECK(NewestAliasTickHolds(images, b));
}
