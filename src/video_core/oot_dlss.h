// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <map>
#include <optional>

#include "common/common_types.h"
#include "common/settings.h"
#include "video_core/surface.h"
#include "video_core/textures/texture.h"

// NVIDIA DLSS in place of The Legend of Zelda: Ocarina of Time's own anti-aliasing network.
// The game renders its 3D at the output size with no jitter and no motion vectors, so this mode
// supplies both: the viewport of every draw into the main scene depth buffer is offset by a
// sub-pixel Halton jitter, and camera motion vectors are rebuilt from depth and the game's
// current/previous view-projection matrices. EDEN_OOT_DLSS=1 turns it on (off by default).
namespace VideoCommon::OotDlss {

constexpr u64 ProgramId = 0x0400486024DEC000ULL;

inline bool Requested() {
    static const bool requested = [] {
        if (const char* value = std::getenv("EDEN_OOT_DLSS")) {
            return value[0] == '1';
        }
        const auto mode = Settings::values.oot_anti_aliasing.GetValue();
        return mode == Settings::OotAntiAliasing::NvidiaDlss ||
               mode == Settings::OotAntiAliasing::NvidiaDlaa;
    }();
    return requested;
}

/// DLSS output: 4K, with the 3D at 4K (DLAA) or at the game's 2560x1440 (Quality: only the
/// output side is rescaled, see OutputOnlyRescale), picked with the anti-aliasing setting.
/// EDEN_OOT_DLSS_OUTPUT=dlaa|quality overrides it.
inline Settings::OotDlssOutput Output() {
    static const std::optional<Settings::OotDlssOutput> override_output =
        []() -> std::optional<Settings::OotDlssOutput> {
        const char* value = std::getenv("EDEN_OOT_DLSS_OUTPUT");
        if (value == nullptr || value[0] == '\0') {
            return std::nullopt;
        }
        return value[0] == 'd' ? Settings::OotDlssOutput::Dlaa4k
                               : Settings::OotDlssOutput::Quality4k;
    }();
    return override_output.value_or(Settings::values.oot_anti_aliasing.GetValue() ==
                                            Settings::OotAntiAliasing::NvidiaDlaa
                                        ? Settings::OotDlssOutput::Dlaa4k
                                        : Settings::OotDlssOutput::Quality4k);
}

/// Only the images after the 3D (DLSS's output, the HUD, the presented copy) are rescaled.
inline bool OutputOnlyRescale() {
    return Requested() && Output() == Settings::OotDlssOutput::Quality4k &&
           Settings::GetCurrentProgramID() == ProgramId;
}

/// Texture LOD bias added to mipmapped samplers under DLSS, NVIDIA's log2(render / display) - 1
/// (DLSS Programming Guide 3.5): the jittered frames DLSS accumulates resolve more texture detail
/// than one frame at the render size. Quality: -1.585; DLAA: -1. The game's own bias stays
/// under it. EDEN_OOT_LOD_BIAS=<n> replaces it in either mode (0: off).
inline float UpscaleLodBias() {
    if (!Requested() || Settings::GetCurrentProgramID() != ProgramId) {
        return 0.0f;
    }
    static const std::optional<float> override_bias = []() -> std::optional<float> {
        if (const char* value = std::getenv("EDEN_OOT_LOD_BIAS")) {
            return static_cast<float>(std::strtod(value, nullptr));
        }
        return std::nullopt;
    }();
    if (override_bias) {
        return *override_bias;
    }
    return (OutputOnlyRescale() ? std::log2(1.0f / 1.5f) : 0.0f) - 1.0f;
}

/// The 3D's images are rescaled (DLAA at 4K).
inline bool SceneRescaled() {
    return Settings::values.resolution_info.active && !OutputOnlyRescale();
}

/// DLSS on the linear HDR scene colour right before the tonemap pass, as NVIDIA recommends, in
/// place of the tonemapped image at the game's network: the tonemap, bloom and HUD then work on
/// DLSS's result. DLAA only (Quality would need the tonemap pass at the output size).
/// EDEN_OOT_DLSS_HDR=0 keeps DLSS after the tonemap.
inline bool HdrInput() {
    static const bool disabled = [] {
        const char* value = std::getenv("EDEN_OOT_DLSS_HDR");
        return value != nullptr && value[0] == '0';
    }();
    return !disabled && Requested() && !OutputOnlyRescale() &&
           Settings::GetCurrentProgramID() == ProgramId;
}

/// Per-object motion vectors for DLSS (on by default with DLSS; EDEN_OOT_OBJECT_MOTION=0 turns
/// them off). Pipelines built while it is off keep camera motion only.
inline bool ObjectMotionRequested() {
    static const bool disabled = [] {
        const char* value = std::getenv("EDEN_OOT_OBJECT_MOTION");
        return value != nullptr && value[0] == '0';
    }();
    return !disabled && Requested();
}

/// DLSS replaces the game's network (No sharpening is video_core/oot_network.h's).
inline bool ReplacesNetwork() {
    return Requested();
}

/// Fragment shaders of the game's tonemap passes (outdoor log-bloom final, adaptive final): the
/// image they render is the network's tonemapped input.
inline bool IsTonemapShader(u64 hash) {
    // 3ac011bd2d772994: the outdoor tonemap at another time of day (Kokiri Forest, evening).
    return hash == 0xbded5ce917183214ULL || hash == 0x6cb06edf14be1d79ULL ||
           hash == 0x3ac011bd2d772994ULL;
}

/// The fullscreen vertex shader of the tonemap passes. The fragment shader varies with the scene
/// (96d6efdbfe52605c in front of the Kokiri shop): matched on its own, an unknown variant kept
/// the last tonemapped image, which froze the copy and fed DLSS a stale frame (smearing).
inline bool IsTonemapVertexShader(u64 hash) {
    return hash == 0x7bb354754112ad1aULL;
}

/// The full-screen draw that composites the half-res volumetric fog over the scene colour. The
/// fireflies, dust and fairy glow are drawn straight into the scene colour after it.
inline bool IsFogCompositeShader(u64 hash) {
    return hash == 0x3ddfb1d7a854c849ULL;
}

/// Where DLSS runs on the HDR path (EDEN_OOT_DLSS_BEFORE): before the particles (default, right
/// after the fog composite), before the fog composite, or at the tonemap. What is drawn after it
/// skips DLSS's history, so particles leave no trails.
enum class HdrStage { Particles, Fog, Tonemap };

inline HdrStage HdrDlssStage() {
    static const HdrStage stage = [] {
        const char* value = std::getenv("EDEN_OOT_DLSS_BEFORE");
        const std::string_view name = value != nullptr ? value : "";
        return name == "fog" ? HdrStage::Fog : name == "tonemap" ? HdrStage::Tonemap
                                                                  : HdrStage::Particles;
    }();
    return stage;
}

/// Why a pipeline that draws into a Z32 depth got no object motion, by its geometry (vertex or
/// mesh) and fragment program hashes: logged when it then draws into the scene depth, so new
/// areas show which of their assets fall back to camera motion only.
struct MotionRejects {
    std::mutex mutex;
    std::map<std::pair<u64, u64>, const char*> reasons;
    std::set<std::pair<u64, u64>> reported;
};

inline MotionRejects& Rejects() {
    static MotionRejects rejects;
    return rejects;
}

inline void NoteMotionReject(u64 geometry, u64 fragment, const char* reason) {
    auto& rejects = Rejects();
    std::scoped_lock lock{rejects.mutex};
    rejects.reasons[{geometry, fragment}] = reason;
}

/// The reason for a scene draw without object motion the first time the pair is seen, else null.
inline const char* FirstSceneReject(u64 geometry, u64 fragment) {
    auto& rejects = Rejects();
    std::scoped_lock lock{rejects.mutex};
    if (!rejects.reported.insert({geometry, fragment}).second) {
        return nullptr;
    }
    const auto it = rejects.reasons.find({geometry, fragment});
    return it != rejects.reasons.end() ? it->second : "not instrumented (pipeline built before)";
}

/// Learned each frame from the network's own texture handles; 0 until seen.
struct SceneTargets {
    std::atomic<u64> depth_gpu{};  ///< D32 main scene depth (jittered draws bind it)
    std::atomic<u64> color_gpu{};  ///< B10G11R11 main scene colour (DLSS input)
    std::atomic<u64> output_gpu{}; ///< B10G11R11 network output (DLSS output)
    std::atomic<u64> tonemapped_gpu{}; ///< B10G11R11 tonemapped scene (the network's input)
    /// Its render target's shape: width, height and block layout (block width | height << 4 |
    /// depth << 8, bit 12 pitch linear), as the tonemap draw bound it.
    std::atomic<u32> tonemapped_width{};
    std::atomic<u32> tonemapped_height{};
    std::atomic<u32> tonemapped_layout{};
    /// Size of the scene depth: every draw into a D32 target of this size is jittered (the game
    /// keeps a second depth of the same size that its colour passes test against).
    std::atomic<u32> depth_width{};
    std::atomic<u32> depth_height{};
    std::mutex depth_mutex;
    std::optional<Tegra::Texture::TICEntry> depth_tic; ///< guarded by depth_mutex
    /// The network output's descriptor (guarded by depth_mutex): the scene colour and the
    /// tonemapped image share its size, format and layout.
    std::optional<Tegra::Texture::TICEntry> output_tic;
    /// Scene frame whose DLSS ran before the tonemap (HdrInput): its network pass only copies.
    std::atomic<u64> hdr_frame{~u64{0}};
    /// The tonemap's exposure (pixel c[3] float 13, written by the CPU), from the last tonemap
    /// draw; 0 until seen. DLSS on the HDR scene colour gets it instead of its auto exposure.
    std::atomic<f32> exposure{};
    /// DLSS after the tonemap: the scene frame whose scene colour was copied before the
    /// particles (after the fog composite; later draws are not jittered), and the one whose
    /// tonemap was drawn again from that copy. DLSS then upscales the image without the
    /// particles, and what they added to the tonemapped image goes on top of its output.
    std::atomic<u64> particle_base_frame{~u64{0}};
    std::atomic<u64> particle_frame{~u64{0}};
};

/// Why DLSS did not run on a frame (it then copies the tonemapped image), counted for a summary
/// every 600 frames: a frame that falls back is never silent.
struct Fallbacks {
    std::mutex mutex;
    std::map<std::string_view, u32> counts;
};

inline Fallbacks& FallbackTally() {
    static Fallbacks fallbacks;
    return fallbacks;
}

inline void NoteFallback(std::string_view reason) {
    auto& tally = FallbackTally();
    std::scoped_lock lock{tally.mutex};
    ++tally.counts[reason];
}

/// "reason xN, ..." since the last call, or empty.
inline std::string TakeFallbackSummary() {
    auto& tally = FallbackTally();
    std::scoped_lock lock{tally.mutex};
    std::string summary;
    for (const auto& [reason, count] : tally.counts) {
        summary += (summary.empty() ? "" : ", ") + std::string{reason} + " x" +
                   std::to_string(count);
    }
    tally.counts.clear();
    return summary;
}

/// Frame counter of the last DLSS evaluation, for HistoryReset.
inline std::atomic<u64>& LastDlssFrame() {
    static std::atomic<u64> frame{~u64{0}};
    return frame;
}

inline SceneTargets& Targets() {
    static SceneTargets targets;
    return targets;
}

/// The images after the 3D in Quality mode: the network output DLSS writes (and the HUD draws
/// on) and the 10-bit copy the game presents.
inline bool IsOutputImage(u64 gpu_addr, VideoCore::Surface::PixelFormat format, u32 width) {
    const u64 output = Targets().output_gpu.load(std::memory_order_relaxed);
    return (output != 0 && gpu_addr == output) ||
           (format == VideoCore::Surface::PixelFormat::A2B10G10R10_UNORM && width >= 1280);
}

/// Radical inverse in `base` (Halton sequence), in [0, 1).
inline float Halton(u32 index, u32 base) {
    float result = 0.0f;
    float fraction = 1.0f / static_cast<float>(base);
    while (index > 0) {
        result += static_cast<float>(index % base) * fraction;
        index /= base;
        fraction /= static_cast<float>(base);
    }
    return result;
}

/// Sub-pixel jitter in render pixels for frame `frame`, in [-0.5, 0.5): Halton (2, 3), 16 phases.
/// EDEN_OOT_DLSS_JITTER_SIGN=-1 flips it for sign checks.
inline std::array<float, 2> Jitter(u64 frame) {
    static const float sign = [] {
        const char* value = std::getenv("EDEN_OOT_DLSS_JITTER_SIGN");
        return value != nullptr && value[0] == '-' ? -1.0f : 1.0f;
    }();
    // EDEN_OOT_DLSS_JITTER=0 keeps the raster still (for A/B and diagnosis).
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_OOT_DLSS_JITTER");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled) {
        return {0.0f, 0.0f};
    }
    const u32 index = static_cast<u32>(frame % 16) + 1;
    return {sign * (Halton(index, 2) - 0.5f), sign * (Halton(index, 3) - 0.5f)};
}

/// The jitter as DLSS and the motion vector pass are told it (the raster takes Jitter).
/// EDEN_OOT_DLSS_REPORT_JITTER=x|y|xy reports those axes the other way round (diagnosis).
inline std::array<float, 2> ReportedJitter(u64 frame) {
    static const std::array<float, 2> signs = [] {
        const char* value = std::getenv("EDEN_OOT_DLSS_REPORT_JITTER");
        const std::string_view axes = value != nullptr ? value : "";
        return std::array{axes.find('x') != std::string_view::npos ? -1.0f : 1.0f,
                          axes.find('y') != std::string_view::npos ? -1.0f : 1.0f};
    }();
    const auto jitter = Jitter(frame);
    return {signs[0] * jitter[0], signs[1] * jitter[1]};
}

/// Row-major 4x4 inverse (general cofactor expansion); nullopt when singular.
inline std::optional<std::array<float, 16>> Inverse(const std::array<float, 16>& m) {
    std::array<float, 16> inv{};
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (!std::isfinite(det) || std::abs(det) < 1e-20f) {
        return std::nullopt;
    }
    for (float& value : inv) {
        value /= det;
    }
    return inv;
}

inline std::array<float, 16> Multiply(const std::array<float, 16>& a,
                                      const std::array<float, 16>& b) {
    std::array<float, 16> result{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a[row * 4 + k] * b[k * 4 + column];
            }
            result[row * 4 + column] = sum;
        }
    }
    return result;
}

/// This frame's reprojection (previous clip from current NDC), captured at the first draw into
/// the scene depth from the game's scene constants: view-projection at +0x30 and the previous
/// frame's at +0x180 of the vertex stage's constant buffer 3, row-major (clip = M * position).
struct Camera {
    std::mutex mutex;
    std::array<float, 16> reproject{};
    /// The current view-projection the reprojection came from (row-major, column vectors).
    std::array<float, 16> view_projection{};
    u64 frame = ~u64{0};
    bool valid{};
    u32 searched{}; ///< draws searched this frame
    bool found{};   ///< this frame's view-projection was found
    bool moving{};  ///< it differs from last frame's
    /// Last frame's view-projection, when it was found (the previous one for camera motion).
    std::array<float, 16> last_view_projection{};
    bool has_last{};
    /// Where the camera's pair was found last (constant buffer, this frame's offset then, and the
    /// other matrix's).
    u32 known_index = ~0u;
    u32 known_offset{};
    u32 known_partner{};
};

/// A row-major view-projection with a perspective divide: clip w is minus the view-space z
/// (a unit rotation row), and clip z is a multiple of it plus a constant.
inline bool IsPerspective(const float* m) {
    for (int i = 0; i < 16; ++i) {
        if (!std::isfinite(m[i])) {
            return false;
        }
    }
    const auto length = [](const float* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    const float w_length = length(m + 12);
    const float z_length = length(m + 8);
    if (w_length < 0.5f || w_length > 2.0f || z_length < 1e-3f || length(m) < 1e-2f ||
        length(m + 4) < 1e-2f) {
        return false;
    }
    const float cosine = (m[8] * m[12] + m[9] * m[13] + m[10] * m[14]) / (w_length * z_length);
    return std::abs(cosine) > 0.9999f;
}

inline Camera& CurrentCamera() {
    static Camera camera;
    return camera;
}

/// Frame counter advanced once per presented frame (gaps reset DLSS's history).
inline std::atomic<u64>& FrameCounter() {
    static std::atomic<u64> frame{};
    return frame;
}

/// Gain from the effects layer's brightest channel to DLSS's bias-to-current-colour mask
/// (fireflies ~0.1 bright make ~0.8). EDEN_OOT_DLSS_MASK_GAIN=<n>; 0 turns the mask off.
inline float MaskGain() {
    static const float gain = [] {
        if (const char* value = std::getenv("EDEN_OOT_DLSS_MASK_GAIN")) {
            return static_cast<float>(std::strtod(value, nullptr));
        }
        return 8.0f;
    }();
    return gain;
}

/// Gain from the volumetric fog layer's luminance to the same mask, off by default: its drag
/// came from the motion vectors' y (FlipY), and a bias over the ground leaves it unsmoothed.
/// EDEN_OOT_DLSS_FOG_GAIN=<n> (2 is mild) adds it.
inline float FogMaskGain() {
    static const float gain = [] {
        if (const char* value = std::getenv("EDEN_OOT_DLSS_FOG_GAIN")) {
            return static_cast<float>(std::strtod(value, nullptr));
        }
        return 0.0f;
    }();
    return gain;
}

/// The image's first row is NDC y = +1 (the game's clip space points y up): the motion vectors
/// flip y. Measured with EDEN_OOT_MOTION_CHECK on a Kokiri Forest pan: predicting each frame
/// from the last with the flip left 0.0087 mean luminance error, without it 0.0155 (no motion
/// 0.0208); unflipped, every vertical camera motion smeared fog and foliage edges.
/// EDEN_OOT_DLSS_FLIP_Y=0 turns it off.
inline bool FlipY() {
    static const bool flip = [] {
        const char* value = std::getenv("EDEN_OOT_DLSS_FLIP_Y");
        return value == nullptr || value[0] != '0';
    }();
    return flip;
}

/// Whether this evaluation starts DLSS's history again: a gap (menu, load, cutscene cut). A
/// present without a new scene (frame pacing at low frame rates) is not a gap.
inline bool HistoryReset() {
    const u64 frame = FrameCounter().load(std::memory_order_relaxed);
    const u64 last = LastDlssFrame().exchange(frame, std::memory_order_relaxed);
    return last == ~u64{0} || frame > last + 8;
}

/// Scene frame: advanced in the GPU command stream right after the network output pass (where
/// DLSS runs), so all of a frame's 3D draws share one jitter phase, one camera capture and one
/// object motion history half. Presents can land while the next frame's draws are being
/// processed, so the present counter above changed in the middle of a frame.
inline std::atomic<u64>& SceneFrame() {
    static std::atomic<u64> frame{};
    return frame;
}

} // namespace VideoCommon::OotDlss
