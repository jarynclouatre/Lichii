// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <source_location>
#include <span>

#include "common/common_types.h"
#include "video_core/buffer_cache/upload_order.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCommon {
struct BufferCopy;
}

namespace Vulkan {

class Scheduler;

/// TEST (EDEN_NX2_UPLOAD_REORDER_CHECK=1): checks the place of every upload from the stream
/// buffer against an exact model of which bytes of which buffers the work recorded in each
/// submission uses. The model is kept apart from the buffers' own tracking (SubmissionUses): it
/// hears of every use the runtime records (descriptor, texel, vertex, index and transform
/// feedback bindings, multi-range sources, copies, clears) and of every use the buffer cache
/// marks (Buffer::MarkUsage), byte for byte. It counts and logs:
/// - an upload placed in the upload command buffer although work recorded before it in the same
///   submission uses a byte it writes (it would run before that work);
/// - one placed there without the barrier on earlier submissions' work although such work, not
///   known finished, uses a byte it writes.
/// Both must stay 0.
class UploadReorderCheck {
public:
    enum class Source : u8 {
        Marked,
        Descriptor,
        Texel,
        Vertex,
        Index,
        TransformFeedback,
        MultiRange,
        CopyRead,
        CopyWrite,
        Clear,
        Count,
    };

    /// Null when the variable is unset.
    [[nodiscard]] static std::unique_ptr<UploadReorderCheck> Create(Scheduler& scheduler);

    ~UploadReorderCheck();

    /// The check in use, null when off.
    [[nodiscard]] static UploadReorderCheck* Active() noexcept {
        return active;
    }

    /// Work recorded now uses [offset, offset + size) of buffer.
    void NoteUse(VkBuffer buffer, u64 offset, u64 size, Source source);

    /// An upload from the stream buffer writing `copies` of buffer is recorded with `order`.
    void CheckUpload(VkBuffer buffer, std::span<const VideoCommon::BufferCopy> copies,
                     VideoCommon::UploadOrder order, std::source_location site);

private:
    explicit UploadReorderCheck(Scheduler& scheduler);

    struct Impl;
    std::unique_ptr<Impl> impl;

    static inline UploadReorderCheck* active = nullptr;
};

} // namespace Vulkan
