# Ocarina of Time: per-object motion vectors for DLSS (implementation plan)

Status: **implemented (milestones 1-3 in one go), being validated.** See "Implementation notes"
at the end for what differs from the plan below and what the test runs showed. The user chose the
"Per-vertex (proper)" route. This plan came from a planning pass over HEAD `8877802878`, so line
numbers are approximate. Background is in `docs/nx2/session-state.md` and
`docs/nx2/oot-network-bypass.md`.

## Idea

Each eligible scene vertex shader stores its clip position into a ping-pong history SSBO, at a slot
keyed by draw identity plus vertex and instance index. It reads last frame's position from the same
slot. Both clip positions go to the fragment shader as smooth varyings. After discard and alpha test,
the fragment shader writes `prev_ndc - cur_ndc` into a motion image. The existing compute pass then
merges object motion with camera motion before DLSS.

Every resource goes through the existing descriptor machinery, as extra bindings appended at the end
of each stage's binding list, so the host and the shader agree on order. Do not use buffer device
addresses: the SPIR-V emitter never sets an addressing model, so they would need memory-model work.

## Findings that shape the design

- **A. Fragments behind other geometry write too.** A fragment shader that stores to an image runs
  before the depth test unless it declares `EarlyFragmentTests`. That is only set from
  `key.state.early_z` (`vk_pipeline_cache.cpp:334` → `emit_spirv.cpp:379`).
  - Each write is therefore tagged with its depth: `uvec2(packHalf2x16(mv), floatBitsToUint(gl_FragCoord.z))`
    into an `R32G32_UINT` image.
  - The combine pass uses an object vector only when that z matches the scene depth (small
    epsilon) and the scene depth is < 1.0. Otherwise it uses camera motion.
  - Later upgrade: a 64-bit atomic-min on an `R64_UINT` image (depth in the high word) so the
    nearest fragment always wins.
  - Perf risk: a storage write without early tests turns early-Z off for every scene pipeline.
    Measure it. If the fragment shader neither discards nor writes depth (`Info.stores_frag_depth`
    false, no demote), it is safe to force `EarlyFragmentTests`.
- **B. Discarded fragments are already excluded.** An `Epilogue` is inserted before every return
  (`structured_control_flow.cpp:851`), and combined demotes sit before the epilogue block
  (`:925-990`). A store placed in `EmitEpilogue` after `AlphaTest` (`emit_spirv_special.cpp:144-156`)
  therefore only runs for kept fragments, and stores from helper invocations have no effect.
- **C. Jitter needs no special handling.** Jitter is applied as a viewport shift
  (`vk_rasterizer.cpp:3138-3194`), so `gl_Position` itself is unjittered. Interpolating cur and prev
  clip positions at the same barycentrics gives
  `mv = 0.5 * (prev.xy/prev.w - cur.xy/cur.w)` in UV units. Negate y when `flip_y` is set. This is
  the same convention as `oot_motion_vectors.comp`.
- **D. Free varying locations.** Generic varyings use locations 0-31 (`DefineInputs` 1768-1829,
  `DefineOutputs` 1998), and NVIDIA allows 128 output components, so there is nothing above 31.
  - Use the two highest locations where `!vs.info.stores.Generic(i)`. The fragment shader only
    declares inputs the vertex shader stores (loop at 1771), so these never collide.
  - If fewer than two are free, or legacy varyings are used, do not inject into that pipeline.

## Steps

1. **Pipeline key bit**
   - Add `BitField<20,1,u32> oot_object_motion` to `fixed_pipeline_state.h:222-234` (raw2 bits
     20-31 are free).
   - Set it in `FixedPipelineState::Refresh` (~138-146) when all of these hold:
     `OotDlss::Requested()`, the program id matches, `regs.zeta_enable`, zeta format is
     `Z32_FLOAT`, and `rt[0]` is B10G11R11.
   - Size is not in the key; step 7 checks it per draw.
   - The bit only reads registers, so disk-cached keys stay deterministic and no cache version bump
     is needed. When loading, skip keys that have the bit while `!Requested()` (loader near
     `vk_pipeline_cache.cpp:1218`).
   - Under `EDEN_NX2_PIPELINE_CLEAN`, also add any new register read to `BuildCleanWatch` (~824).
2. **Per-pipeline policy**
   - Add to `Shader::Info` (`shader_info.h`, near `uses_rescaling_uniform` :364):
     `struct { bool history, image; u32 loc_cur, loc_prev; } oot_motion`.
   - Fill it in `PipelineCache::CreateGraphicsPipeline` after the translate loop
     (`vk_pipeline_cache.cpp:1724-1786`), before `MakeRuntimeInfo` and emission (1830). The
     deferred/async emission path copies the programs, so the flags reach it.
   - Inject only when all of these hold:
     - the key bit is set;
     - the pipeline is VertexB + Fragment only (`unique_hashes[2..4] == 0`, no
       `requires_layer_emulation`), and has no task or mesh stage;
     - `xfb_enabled == 0` and `!uses_render_area`;
     - there are no legacy varyings and two locations are free;
     - the fragment shader does not store depth (or the written depth is used for the tag).
   - The vertex stage gets `history = true` and the fragment stage gets `image = true`.
3. **SPIR-V resources** (`spirv_emit_context.cpp`, constructor 473-507)
   - Add `DefineOotMotion(program, binding)` right after `DefineImages` (499), using the same
     binding counter.
   - Vertex stage: an SSBO `vec4 history[]`, and `cur`/`prev` outputs at the chosen locations. Also
     define its own `VertexIndex`/`InstanceIndex` builtins if they don't exist yet (1743-1753).
   - Fragment stage: a writeonly `rg32ui` storage image, and smooth inputs at the same locations.
4. **Push constants** (`emit_spirv.h:25-45`)
   - Add `struct OotMotionLayout { u32 cur_base, prev_base, vertex_bias, vertex_range, instance_bias, flags, pad[2]; }`
     at `OOT_MOTION_OFFSET = 48`, after Rescaling (32 bytes) and RenderArea (16 bytes).
   - Only one push-constant block per entry point is allowed, so append these members in
     `DefineRescalingInputPushConstant` (1181) and raise its `static_vector<Id,4>` to 10. If no
     block exists, define a standalone one following `DefineMeshDrawBase` (1260).
   - Grow the range in `DescriptorLayoutBuilder::CreatePipelineLayout`
     (`pipeline_helper.h:245-268`) to 80 bytes. That is within the 128-byte minimum.
   - Existing quirk: RenderArea sits at offset 0 and overlaps Rescaling. Leave it, since injection
     is already off when RenderArea is used.
5. **Shader code** (new `emit_spirv_oot_motion.cpp`, called from `EmitEpilogue`)
   - Vertex stage, before `ConvertDepthMode` (which only changes z):
     ```
     pos = load(output_position)
     prev = pos
     if (flags & 1) {
       l = VertexIndex - vertex_bias; i = InstanceIndex - instance_bias
       if (l < vertex_range) {
         s = i * vertex_range + l
         history[cur_base + s] = pos
         prev = (prev_base != ~0u) ? history[prev_base + s] : pos
       } else prev.w = 0          // marks "no history"
     }
     out_cur = pos; out_prev = prev   // always written
     ```
   - Fragment stage, after `AlphaTest`, when `flags & 1`, `prev.w != 0` and
     `!IsHelperInvocation`:
     - compute `mv`;
     - drop it if `|mv| >= 0.25`, which is treated as garbage;
     - `imageStore(img, ivec2(gl_FragCoord.xy), uvec2(packHalf2x16(mv), floatBitsToUint(gl_FragCoord.z)))`.
   - Under resolution scaling, the raw host `FragCoord` is in scaled pixels. That matches a scaled
     motion image, and the IR rescaling pass does not touch backend-injected reads.
6. **Host layout and descriptors**
   - `NumDescriptorEntries` (`pipeline_helper.h:137`): add 1 when the flag is set.
   - `DescriptorLayoutBuilder::Add(info, stage)` (:270-280): append the STORAGE_BUFFER (vertex)
     or STORAGE_IMAGE (fragment) binding after `Add(STORAGE_IMAGE)`. This keeps it per stage and
     at the tail, matching step 3.
   - In `GraphicsPipeline::ConfigureImpl`'s `prepare_stage` (`vk_graphics_pipeline.cpp:738-773`)
     and in `ConfigureDelta` (829-847), after `PushImageDescriptors`, call
     `guest_descriptor_queue.AddBuffer(history, 0, size)` or `AddImage(motion_view)`.
   - The count check in `ConfigureDraw` (873) fails loudly on any ordering mistake.
   - Push-descriptor overflow (`CanUsePushDescriptor` :150) already falls back by itself.
   - Do not set `writes_storage`/`writes_images` (constructor 312-321). That leaves fast-draw
     eligibility (`NoteResolvedDraw` :376) unchanged.
7. **Per-draw constants**
   - In `RasterizerVulkan::PrepareDraw`, keep the pipeline in a member before `draw_func()`
     (`vk_rasterizer.cpp:910`).
   - In `Draw`'s lambda (1207-1265), right before the draw is recorded, call
     `oot_object_motion.Push(...)` with the `DrawParams` from `MakeDrawParams` (225-246). It records
     `cmdbuf.PushConstants(layout, push_stages, 48, 32, &data)`.
   - This covers the Full, Delta and Reused fast-draw paths: Reused skips Configure, but
     `draw_func` always runs, and the descriptors are constant.
   - Indirect draws (1268), mesh draws and byte-count draws must still push `flags = 0`, because
     push constants persist across draws.
   - A draw is enabled by the same size + format test as `OotDlssViewportJitter` (~3120).
8. **Slot allocator** (new `vk_oot_object_motion.{h,cpp}`, owned by `RasterizerVulkan`)
   - Storage: one device-local SSBO of 2 × capacity `vec4`, for example 4M slots (64 MB per half),
     ping-ponged by frame parity.
   - Draw identity hash:
     - VS/FS hashes (`ShaderHashes()`);
     - index buffer address, format, first and count (or first vertex and count when not indexed);
     - instance count and vertex strides;
     - an occurrence counter within the frame, for the same mesh drawn N times.
   - Vertex-buffer and cbuf addresses are tie-breakers only. Compute-skinned ring buffers and
     per-frame uniform rings change them every frame.
   - `cur_frame` and `prev_frame` are `unordered_map<u64, Slot{base, range, instances}>`, filled by
     a bump allocator. Reuse `prev_base` only if range and instances match. On overflow, set
     `flags = 0`.
   - `BeginFrame()`, called from `TickFrame` (~2506):
     - swaps the maps;
     - records a buffer barrier from VERTEX_SHADER write to read, outside the render pass. The
       barrier inside `Generate` does not run in menus.
   - Log per-frame stats: injected draws, matched draws, slots used and overflow.
9. **Index formula**
   - The emulator's VertexId is raw `gl_VertexIndex`, which includes base vertex / first vertex
     (`emit_spirv_context_get_set.cpp:354-358`). Push the biases instead of using the `BaseVertex`
     builtin, which avoids the DrawParameters capability.
   - Non-indexed draws: `vertex_bias = first`, `range = count`.
   - Indexed draws:
     - `vertex_bias = vertexOffset` (`draw_state.base_index`), so `l` is the raw index;
     - `range = max_index + 1`, from a CPU scan of the guest index buffer
       (`gpu_memory->ReadBlockUnsafe`), cached by (address, format, first, count) and skipping the
       restart index;
     - if the scan isn't available, fall back to the vertex-stream limit, clamped.
   - Instances: `instance_bias = base_instance`.
   - Quads and quad strips are rebuilt as indexed draws with `base_vertex = 0` (236-244). Leave
     them disabled at first.
10. **Combine pass**
    - Shader changes: add binding 2 (the `rg32ui` object image) to `oot_motion_vectors.comp`.
    - Host changes: give `vk_oot_motion_vectors.cpp` a three-binding layout, and have `Generate`
      take the image view.
    - Per pixel: use the object vector if its stored z ≈ depth and depth < 1, else camera motion.
      Then write back `uvec2(0, ~0u)` so the image is clear for the next frame. Also clear it once at
      creation.
    - Create the image at the scene framebuffer's render-area size (scaled under resolution
      scaling), and recreate it if that size changes.
    - Pass it from `TextureCacheRuntime::OotDlss` (`vk_texture_cache.cpp:2003-2008`). The existing
      barrier in `Generate` covers fragment write → compute read.

## Pitfalls

- **Pipelines shared with non-scene draws:** the key covers formats only, so off-size draws must
  push `flags = 0`.
- **Sky and fullscreen passes** with constant clip positions write zero motion. The depth < 1 rule
  handles them, plus an optional per-hash denylist.
- **Depth prepass:** it has no colour target, so it isn't keyed. Only the colour pass writes
  history.
- **Shader cache and async compile:** SPIR-V is never stored on disk, only keys and environments,
  so the caches need no change. The async emission worker and `EDEN_NX2_PERF_VERIFY` reuse the same
  flags.
- **Frame skip (`Nx2FrameSkip`):** a skipped frame must not swap the maps.
- **Vertex repack (`AnyVertexRepacked`):** check that vertex indices are unchanged.
- **Mesh shaders (foliage), later:**
  - Hook in `MeshEpilogue` (`emit_spirv_mesh.cpp:120`). After the barrier, each invocation walks
    `v = local; v < count; v += wg`, reads `output_position[v]` (array at
    `spirv_emit_context.cpp:1925`) and uses slot `base + WorkgroupId.x * max_vertices + v`.
  - This needs two extra per-vertex output arrays.
  - Workgroup ids are unstable when a task stage culls, so this is deferred and foliage keeps
    camera motion.
- **Indirect draws:** counts are not known on the CPU (`DrawIndirect` 1268), so these are
  deferred. A later option is to read the arguments on the CPU when `!IsRegionGpuModified`.
- **Transform feedback, geometry and tessellation:** excluded by the step 2 policy.

## Milestones

1. **First milestone**
   - Scope:
     - vertex + fragment pipelines only;
     - non-indexed, non-indirect draws with triangle topologies;
     - history and flags as above;
     - the `R32G32_UINT` image with the depth check;
     - a debug mode that outputs the object-motion mask in place of the motion vectors.
   - Test in Kokiri Forest: hold the camera still while NPCs or Link's idle animation move. Drive
     the game with `tools/nx2/oot_autoload.py` and the debug server's `input_sequence`.
   - Pass criteria:
     - from the second frame on, object pixels match in ≥ 90 % of injected draws (combine log);
     - the mask covers characters;
     - ghosting is gone on moving objects;
     - while panning, object vectors agree with camera motion on static geometry within ~0.1 px;
     - fps is not far below today's ~42-46.
2. **Indexed draws**, using the cached index scan. Most of OoT's characters are indexed, so expect
   the real gains to start here.
3. **Precision and performance:** the R64 atomic-min, and forcing early tests where safe.
4. **Mesh-shader pipelines** (foliage).
5. **Indirect draws.**

## Critical files

- `src/shader_recompiler/backend/spirv/spirv_emit_context.cpp`
- `src/shader_recompiler/backend/spirv/emit_spirv_special.cpp`
- `src/shader_recompiler/backend/spirv/emit_spirv.h`
- `src/shader_recompiler/shader_info.h`
- `src/video_core/renderer_vulkan/fixed_pipeline_state.{h,cpp}`
- `src/video_core/renderer_vulkan/vk_pipeline_cache.cpp`
- `src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp`
- `src/video_core/renderer_vulkan/pipeline_helper.h`
- `src/video_core/renderer_vulkan/vk_rasterizer.cpp`
- `src/video_core/renderer_vulkan/vk_oot_motion_vectors.{h,cpp}`
- `src/video_core/host_shaders/oot_motion_vectors.comp`
- new: `emit_spirv_oot_motion.cpp`, `vk_oot_object_motion.{h,cpp}`

## Implementation notes

What was built (files: `vk_oot_object_motion.{h,cpp}`, `emit_spirv_special.cpp`
`OotMotionVertex`/`OotMotionFragment`, `spirv_emit_context.cpp` `DefineOotMotion*`,
`vk_pipeline_cache.cpp` policy block, `vk_rasterizer.cpp` `PushOotObjectMotion`/`OotIndexBounds`,
`oot_motion_vectors.comp`). Differences from the plan:

- **No pipeline key bit.** The policy reads `key.state.depth_enabled`/`depth_format` (Z32) and
  `ObjectMotionRequested()` (DLSS mode, `EDEN_OOT_OBJECT_MOTION=0` turns it off). Toggling needs a
  restart; SPIR-V is not cached on disk, so nothing stale survives.
- **Per-draw gate** (`PushOotObjectMotion`): scene-size Z32 depth, depth test **and depth write**
  on, not quads. Without the depth-write rule, fullscreen fog/lighting passes covered every pixel.
- **Every graphics pipeline layout's push range is 80 bytes**, so push constants stay compatible
  across pipelines (the down factor is only pushed when it changes).
- **Indexed draws are in from the start**: `OotIndexBounds` scans the guest index range on the CPU
  (cached by start, count, format, restart index; up to 16 MiB) and the draw's range is
  `max - min + 1` with `vertex_bias = base_vertex + min`. Estimating from the vertex stream size
  overflowed the 4M-slot history (shared vertex buffers).
- **Nearest fragment wins via a 64-bit buffer atomic** (not an R64 image): the fragment stage
  `atomicMin`s `depth_bits << 32 | packHalf2x16(motion)` into a per-pixel `u64` buffer
  (needs `shaderBufferInt64Atomics`, on everywhere but Qualcomm). With plain stores, 82% of pixels
  ended up holding a fragment *behind* the visible surface. Non-discarding fragment stages also get
  `EarlyFragmentTests` (`oot_motion.early_tests`).
- **Motion buffer resizes only in `TickFrame`**: fast "reused" draws keep last draw's descriptors,
  so the buffer must not change within a frame.
- **Tonemap detection** gained fragment shader `3ac011bd2d772994` (outdoor tonemap at another time
  of day); with only the two old hashes, DLSS mode silently fell back to the game's network.
- `EDEN_OOT_OBJECT_MOTION_STATS=<n>` logs, every n DLSS evaluations (keep n small at 4K: the sums are 32-bit): share of scene pixels with a fragment
  write, share using it, agreement with camera motion (y as is / negated), and writes behind / in
  front of the scene depth. The atomics cost ~15 fps; leave it off for benchmarks.

Results so far (Kokiri Forest, 1x, RTX 5090):

| Run | fps | Notes |
|---|---|---|
| Object motion off | 41.9 | camera motion only |
| Object motion on (plain stores) | 43.6 | no measurable cost |
| History | — | ~1,650 draws/frame, 99% find last frame's slots, ~390K slots/frame, no overflow; ~53% of draws repeat an identity (same mesh drawn again), told apart by occurrence order |

### Later fixes (same session)

- **Frame boundary in the command stream.** History swap, jitter phase and camera capture used
  the present counter (`TickFrame`), which advanced while the next frame's draws were being
  processed: part of a frame's 3D got one jitter phase and the rest another, and draws after the
  swap read "last frame" positions written earlier in the same frame (zero motion on pans).
  `OotDlss::SceneFrame()` now advances right after the network output pass (DLSS or its copy),
  and `RasterizerVulkan::OotSceneEnd` swaps the history there. DLSS history resets need a gap of
  more than 8 presents (was 2, which tripped on ordinary frame pacing).
- **Repeated meshes get no history.** A mesh drawn more than once a frame was matched by draw
  order, which the game does not keep; copies traded histories. They now fall back to camera
  motion (`EDEN_OOT_OBJECT_MOTION_REPEATED=1` restores the old matching).
- **Camera matrices are searched for.** The fixed offsets (VS c[3] +0x30 / +0x180) only held for
  some shaders: at other times of day the first scene draw's c[3] held a plain view matrix there
  and constant data, which gave near-zero camera motion on pans and a wedge of huge vectors.
  `CaptureOotCamera` now scans the vertex stage's constant buffers for two perspective
  view-projections (`OotDlss::IsPerspective`) that are close, prefers a pair that differs
  (seen: c[7] +0xc0 current, +0x100 previous; c[3] +0xa0 / +0x270 holds the same matrix twice)
  and searches up to 64 draws a frame for it.
- **4K output.** `oot_anti_aliasing` NVIDIA DLSS (4K Quality from 1440p) or NVIDIA DLAA (4K). Quality rescales
  only the output chain (`OotDlss::IsOutputImage`: the network output and the A2B10G10R10 present
  copy) at 1.5x, so DLSS runs 2560x1440 -> 3840x2160 (mode 3). DLAA rescales everything and
  DLSS runs 3840x2160 -> 3840x2160 (mode 6). It used to wash out white: the scene depth is a
  33-layer D32 array (274M texels at 1.5x), which the 2^28-texel guard in `ImageCanRescale` kept
  at 1x, and every colour target bound with it fell back to 1x pass by pass. The guard now
  follows the allocator's backing rule (`ShouldLeaveImageUnbacked`, Lichii #267 `e15125b73`);
  `EDEN_NX2_RESCALE_TRACE=1` logs why an image stays at 1x. OoT uses preset K
  (`EDEN_NX2_DLSS_PRESET` overrides).
- **Diagnostics:** `EDEN_OOT_MOTION_DEBUG=1` shows the motion vectors in place of the image (red:
  length / 32 px; green 0.6: object motion used; +0.3: empty depth). `EDEN_OOT_OBJECT_MOTION_STATS`
  adds mean |x| of object and camera motion and x direction agreement.

Open: confirm by eye that pans are no longer blurred and the resting view no longer jitters;
quantify camera vs object agreement with a fast, level pan; 4K DLAA; lower DLSS modes through the
game's dynamic resolution.

### Mesh-shader foliage (2026-11-06)

- **Foliage is mesh shaders.** Skipping mesh draws (`EDEN_NX2_NO_MESH_FILE`) removes the canopy,
  bushes, grass and plants. Most scene mesh pipelines have no task stage
  (`EDEN_NX2_MESH_PIPELINE_LOG=1`).
- **History per (draw, workgroup, vertex) does not work for them.**
  `EmitOotMotionMeshVertex` stores position plus an attribute signature per vertex.
  - Signature over all attributes: nothing matches frame to frame.
  - Signature over the first or last attribute only: matches, but the motion is 32+ px with a
    still camera.
  - So a workgroup draws different vertices every frame. Most likely the meshlet list is
    rebuilt by a culling pass each frame.
  - The per-vertex history is kept behind `EDEN_OOT_MESH_MOTION=history` (signature
    selectable with `EDEN_OOT_MESH_SIGNATURE=first|last`).
- **The sound approach: evaluate the vertex/mesh program twice.** Once as is, and once with last
  frame's constant buffers bound in place of this frame's (a recompiler pass that duplicates the
  position computation with its constant-buffer loads redirected to a second binding set). The
  second result is the previous clip position of the same vertex, whatever the culling did, with
  no matching.
  - The host keeps last frame's copy of each constant buffer range a draw reads, keyed like the
    history (per-draw identity; static per-instance data is the same either way, and the global
    time/camera blocks are what change).
  - This would also replace the history buffer for vertex shaders.

### Mesh shadow evaluation (implemented 2026-11-06)

The approach above, done in the mesh stage only (default; `EDEN_OOT_MESH_MOTION=0` turns mesh
motion off).

- **Probe first.** Over 60 scene mesh programs, every position depends on constant buffers 0, 3
  and 4 (sometimes 5 and 6) at immediate binding indices, nothing reaches an attribute write
  through shared memory, and the dependent side effects are only branch conditions, the map
  writes and a few storage writes.
- **`OotShadowPass`** (`ir_opt/oot_shadow_pass.cpp`, run by the pipeline cache):
  - clones the pure instructions (and phis) that depend on immediate constant buffer reads;
  - turns the clones' reads into `OotPrevCbufU32` (last frame's word);
  - adds an `OotShadowWriteAttribute` twin to every attribute write.
  The shadow follows this frame's control flow; memory loads read this frame's data.
- **SPIR-V.**
  - Shadow position components go to a workgroup array.
  - In the epilogue, workgroup 0 copies the constant buffer words the shadow reads into the
    history buffer (four words per slot), and each vertex passes its position plus the shadow's.
  - `OotPrevCbufU32` reads the snapshot at `previous_base`, or this frame's word when there is
    none.
- **Matching.**
  - Constant buffers are ring-allocated, so their addresses change every frame and cannot
    identify a draw (0 matches).
  - Repeated draws of the same mesh are matched by draw order: 94% of scene mesh draws find a
    snapshot (30% without repeats).
  - A shadow position more than 0.1 NDC from the current one is dropped, so a mismatched pair
    falls back to camera motion.
- Task stages are allowed: nothing depends on the workgroup numbering any more.
- **Results (Kokiri Forest, still camera, motion debug view).** Canopy, trees and mid-ground
  plants switch from camera to object motion with near-zero vectors, and foliage renders
  unchanged. Not yet checked by eye: DLSS foliage stability and FG artifacts while moving.
