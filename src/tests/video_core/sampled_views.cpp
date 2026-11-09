// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/slot_vector.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/sampled_views.h"

namespace {

using VideoCommon::ImageBase;
using VideoCommon::ImageFlagBits;
using VideoCommon::ImageId;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::ImageViewBase;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewInfo;
using VideoCommon::ImageViewType;
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

struct View {
    ImageViewId id;
};

/// Images and views laid out as TextureCache makes them: the null image and view first.
struct Cache {
    Cache() {
        REQUIRE(images.insert(VideoCommon::NullImageParams{}) == VideoCommon::NULL_IMAGE_ID);
        REQUIRE(views.insert(VideoCommon::NullImageViewParams{}) ==
                VideoCommon::NULL_IMAGE_VIEW_ID);
    }

    ImageId AddImage() {
        const ImageId id = images.insert(Info(), 0x10000, 0x10000);
        images[id].flags &= ~ImageFlagBits::CpuModified; // uploaded
        return id;
    }

    ImageViewId AddView(ImageId image) {
        const ImageViewInfo info{ImageViewType::e2D, PixelFormat::A8B8G8R8_UNORM};
        return views.insert(info, images[image].info, image, 0x10000);
    }

    bool Change(std::vector<View> sampled) const {
        return VideoCommon::SampledViewsChange(sampled, views, images);
    }

    Common::SlotVector<ImageBase> images;
    Common::SlotVector<ImageViewBase> views;
};

} // namespace

TEST_CASE("The null view is never prepared", "[video_core][texture_cache]") {
    Cache cache;
    // Its image has no valid format and stays CpuModified: preparing it uploads it (#904).
    REQUIRE(True(cache.images[VideoCommon::NULL_IMAGE_ID].flags & ImageFlagBits::CpuModified));
    CHECK_FALSE(VideoCommon::PreparesImage(cache.views, VideoCommon::NULL_IMAGE_VIEW_ID));
    CHECK_FALSE(cache.Change({{VideoCommon::NULL_IMAGE_VIEW_ID}}));

    // A draw sampling an uploaded texture beside a null one changes nothing.
    const ImageViewId view = cache.AddView(cache.AddImage());
    CHECK_FALSE(cache.Change({{VideoCommon::NULL_IMAGE_VIEW_ID}, {view}}));
}

TEST_CASE("Buffer views and empty ids are never prepared", "[video_core][texture_cache]") {
    Cache cache;
    ImageInfo info;
    info.type = ImageType::Buffer;
    info.format = PixelFormat::R32_UINT;
    info.size = {256, 1, 1};
    const ImageViewInfo view_info{ImageViewType::Buffer, PixelFormat::R32_UINT};
    const ImageViewId buffer = cache.views.insert(info, view_info, 0x20000);
    // A buffer view points at the null image.
    REQUIRE(cache.views[buffer].image_id == VideoCommon::NULL_IMAGE_ID);
    CHECK_FALSE(VideoCommon::PreparesImage(cache.views, buffer));
    CHECK_FALSE(VideoCommon::PreparesImage(cache.views, ImageViewId{}));
    CHECK_FALSE(cache.Change({{buffer}, {ImageViewId{}}}));
}

TEST_CASE("A sampled image with an upload pending changes", "[video_core][texture_cache]") {
    Cache cache;
    const ImageId image = cache.AddImage();
    const ImageViewId view = cache.AddView(image);
    CHECK(VideoCommon::PreparesImage(cache.views, view));
    CHECK_FALSE(cache.Change({{view}}));
    cache.images[image].flags |= ImageFlagBits::CpuModified;
    CHECK(cache.Change({{view}}));
}

TEST_CASE("A sampled image changes while a newer alias has not been copied in",
          "[video_core][texture_cache]") {
    Cache cache;
    const ImageId sampled = cache.AddImage();
    const ImageId target = cache.AddImage();
    const ImageViewId view = cache.AddView(sampled);
    REQUIRE(AddImageAlias(cache.images[sampled], cache.images[target], sampled, target));
    VideoCommon::SetModificationTick(cache.images, cache.images[sampled], 4);
    VideoCommon::SetModificationTick(cache.images, cache.images[target], 3);
    CHECK_FALSE(cache.Change({{view}}));

    // A draw renders into the alias (TextureCache::MarkModification): the next sample copies it.
    VideoCommon::SetModificationTick(cache.images, cache.images[target], 5);
    CHECK(cache.Change({{view}}));

    // As SynchronizeAliases leaves it after the copy.
    VideoCommon::SetModificationTick(cache.images, cache.images[sampled], 5);
    cache.images[sampled].newest_alias_tick = 5;
    CHECK_FALSE(cache.Change({{view}}));

    // A newer alias deleted before it was copied leaves nothing to copy.
    VideoCommon::SetModificationTick(cache.images, cache.images[target], 6);
    cache.images[sampled].aliased_images.clear();
    CHECK_FALSE(cache.Change({{view}}));
}
