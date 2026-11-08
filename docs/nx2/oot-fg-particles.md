# Ocarina of Time: particles and DLSS Frame Generation (plan)

## Where things stand (2026-11-07)

- DLSS (both paths) now runs before the particles: fireflies, dust, fluff and fairy glow are
  transparent or additive draws after the fog composite that don't write depth.
  - DLAA (HDR path): DLSS runs at the first such draw; the particles are drawn over its result.
  - Quality (after the tonemap): the tonemap draw is repeated from a copy of the scene colour
    taken before the particles. DLSS upscales that, and what the particles added to the
    tonemapped image (with minus without, at the render size) goes on top of its output.
- So the particles are in neither DLSS's history nor its motion vectors. Frame generation reads
  the same motion vectors: in a generated frame a particle moves with whatever is behind it
  (camera and object motion of the scene), not with itself.
- The HUD already has its own path: `OotUiAlpha` copies DLSS's output before the HUD and tags the
  difference with the finished image as `kBufferTypeUIAlpha`, so FG leaves the HUD alone.

## Expected artifact

Fireflies drift slowly, so most of the time background motion is close enough. Expect visible
problems where the background moves differently from the particle:

- a particle in front of a moving character or swaying foliage gets dragged along with it;
- fast particles (sparks, fairy trails, effects in combat) double or smear between real frames.

## Plan, cheapest first

1. **Treat particles like the HUD.** We already know, per pixel, what the particles added to
   the final image: in Quality the with-minus-without difference of the tonemapped image; in DLAA
   the same if the tonemap redraw is also used there (or scene colour at the tonemap minus the
   copy taken where DLSS ran, tonemapped approximately). Add that coverage to the UI alpha mask
   (`OotUiAlpha::Tag`), so FG shows the particles from the latest real frame unchanged.
   Particles then move at the base frame rate (~50-60 fps) while the scene moves at the
   generated rate. For slow fireflies that is invisible; it is the safe default.
2. **Measure before going further.** With FG on, compare particles with and without step 1, and
   look at fast effects. If base-rate particles look like stutter, go to step 3.
3. **Give particles their own motion.** The particle draws are ordinary vertex draws after DLSS.
   Draw their screen motion into a separate image (the per-vertex history the scene draws already
   use, keyed by draw identity, or the particle system's previous positions if its buffers keep
   them). Combine it into FG's motion vectors where the particle mask says the particle dominates.
   Particle identities may not be stable from frame to frame, so check how many keep their
   history before relying on it.

## Not needed

- No change to DLSS itself: particles stay out of its input as now.
- The tonemap redraw already exists for Quality; reusing it in DLAA only for FG is optional
  (cost: one full-screen pass per frame).
