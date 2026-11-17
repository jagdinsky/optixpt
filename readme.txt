// Use "P" button for switching between Path Tracing and Photon Mapping

// build — Release: every timing in the thesis is a Release build with OptiX
// validation off (the timing JSON records both, and exec.py warns otherwise).
// Measurements use build-release/: the editor's CMake integration reconfigures
// build/ with its own variant (Debug) whenever CMakeLists.txt changes, so
// build/ is for the IDE and build-release/ for numbers.  checks.py, exec.py
// and eval/ use build-release/ when it exists (RENDERER_BUILD=dir overrides)
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc

// compile
cmake --build build-release -j$(nproc)

// run, from the build directory (the scenes are read from ../scenes)
cd build-release
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./renderer

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./renderer scene.glb --offline --camera camera.txt --output output.exr
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./renderer scene.glb --offline --camera camera.txt --output output.exr --photon

// offline options: --frames N   frames to accumulate
//                  --paths N    photon paths per frame (photon mapping only)
//                  --depth N    max path length
//                  --seed N     random seed; 0 (the default) keeps every run
//                               reproducible, and two runs differing only in
//                               the seed are statistically independent, which
//                               is what proof.py measures the noise level from
//                  --res W H    offline resolution (default 1280x720); small
//                               frames are what make a brute-force run cheap

// photon mapping renders one camera sample per frame, always.  Extra samples in
// a frame all query the same photon map at the same radius, so they do nothing
// about the density estimate — the error that actually dominates the image —
// while multiplying the cost of the gather.  Spend the budget on --frames.

// the gather runs as two launches: a camera pass that parks every pixel at its
// first diffuse vertex, and a query pass that does nothing but look photons up.
// Only the second one changes when the acceleration structure changes, so it is
// the one the comparison is about.  The camera and photon passes are OptiX
// programs in one "trace" pipeline; the query pass is a plain CUDA kernel, one
// per structure (src/gather.h), so that each gets its own register allocation
// — OptiX gives every program of a pipeline the same register count, and the
// lookups need no ray tracing.  The timing JSON lists every kernel's registers
// and local memory; the OptiX programs' need Nsight Compute, with GPU counter
// access: sudo ncu --metrics launch__registers_per_thread ./renderer ...
//
// configurations are named <family>/<accel>/<radius mode> in logs and timing
// records: family A gathers from the full photon map, B from a culled one
// (--culling on), e.g. A/hash/global, B/mlhash/footprint
//
// gather options:  --accel S          structure the gather searches with:
//                                       uniform     uniform grid (default;
//                                                   also: grid)
//                                       hash        full hash grid
//                                       stochastic  stochastic hash grid: one
//                                                   photon per bucket, the last
//                                                   one written (Hachisuka &
//                                                   Jensen 2010) — biased
//                                       rect        rectified stochastic hash
//                                                   grid: one photon per bucket,
//                                                   picked by reservoir sampling
//                                                   (Davidovic et al. 2014) —
//                                                   unbiased (also: rectified)
//                                       mlhash      multi-level hash grid
//                                       mlrect      multi-level rectified grid
//                                       brute       no structure; test every photon
//                  every structure visits the cells the lookup's sphere
//                  overlaps (8..27 cells between r and 2r wide, fewer when the
//                  cells are wider), not a fixed 3x3x3 block
//                  --table-load F     every hash table gets the smallest power
//                                     of two of buckets that is at least F per
//                                     entry it holds (default 4); the tables
//                                     filled during the trace (stochastic,
//                                     rect) count the photon capacity as their
//                                     entries
//                  --validation on|off  OptiX validation (default off): extra
//                                     checks and a synchronisation after every
//                                     launch, so every pass time includes them
//                  --brute-force      shorthand for --accel brute
//                  --verify-gather    run the query pass twice per frame, the
//                                     selected structure against brute force,
//                                     and report how far apart the two images
//                                     are (exits non-zero if they disagree).
//                                     Not for the stochastic grids: they only
//                                     estimate the brute-force answer
//
// the stochastic grids are filled by the photon pass itself, as it deposits,
// so they have no build: their build_ms is just clearing the table, and the
// filing is inside photon_trace_ms
//                  --timing-json F    write per-frame, per-pass timings to F
//                  --warmup N         frames left out of the timing means (1)
//
// the realtime GUI has the same choice under Photon map > Acceleration structure
//
// checking a structure against brute force — same photons, same gather points,
// only the lookup differs, so any disagreement is the structure's fault:
//   ./renderer scene.glb --offline --photon --camera camera.txt \
//       --res 320 180 --frames 4 --paths 50000 --accel hash --verify-gather \
//       --output ../build/verify.exr
//
// timing JSON: the configuration (config, family, accel), the environment
// (build type, OptiX validation, GPU, driver), every kernel's registers, the
// table load, and per frame: radius, photons, cells (uniform grid only),
// buckets (hash grids only), and camera_ms, mark_ms (culling only),
// photon_trace_ms, build_ms, query_ms, accum_ms; also deposits, culled, r_max,
// r_mean, r_px_mean, num_levels, level_min, level_max, entries (multi-level
// grids) and bytes (structure memory the lookup reads), and with --stats
// tested/accepted/cells mean, p95, max
//
// the frame order is camera -> photons -> build -> query -> accumulate: the
// camera pass goes first so the photon pass and the build know what this
// frame's lookups need (culling marks, levels, the largest radius)
//
// photon culling:  --culling on|off   store only photons some lookup of the
//                                     frame can reach (default off).  Exact:
//                                     a photon within r of a lookup lies in
//                                     one of the cells it marked — the cells
//                                     its sphere overlaps at its level, the
//                                     same ones the multi-level grids visit.
//                                     The marks are a pass of their own
//                                     (mark_ms: clearing the mask and
//                                     marking), after the camera pass
//                  --mask-bits N      culling mask of 2^N bits (default 24),
//                                     addressed by a hash independent of the
//                                     tables' buckets
//                  --verify-culling   trace every frame twice unculled and
//                                     once culled, compare the lookups
//                                     (not for the one-photon-per-bucket grids)
//
// per-pixel radius: --radius-mode global|footprint
//                                     global: one radius per frame (default);
//                                     footprint: r = s * k_pix * t * pixelAngle,
//                                     t the camera path length to the diffuse
//                                     vertex, s the PPM schedule's factor
//                  --k-pix F          radius in pixel footprints (default 4)
//                  --s0 F             level-0 cell side in world units
//                                     (default scene diagonal * 2^-18);
//                                     level l has cells s0 * 2^l
//                  --debug-exr P      write P.rpx.exr (radius in pixels),
//                                     P.level.exr and, with --stats,
//                                     P.accepted.exr for the last frame
//
// multi-level grids: --accel mlhash   every photon filed once per cell level
//                                     in use; a lookup searches the cells its
//                                     sphere overlaps on its own level (cells
//                                     between r and 2r wide, 8..27 of them)
//                    --accel mlrect   the same, one photon per bucket picked
//                                     by the smallest random key (unbiased)
//                    --rect-buckets N mlrect table size, a power of two
//                                     (default: --table-load per insertion;
//                                     with culling the insertions are counted
//                                     first)
//                    --mlrect-in-trace  the photon pass inserts into mlrect as
//                                     it stores (no kernel afterwards; table
//                                     sized from the photon capacity)
//
// --stats            per lookup: photon records tested and accepted, and cells
//                    visited (mean, p95, max per frame; off by default — the
//                    query kernels without it carry no counters at all)
// --first-frame N    start at frame index N: frame i is the same frame in
//                    every run, so this renders exactly frames N.. of a long run
// --float-exr        32-bit float output instead of half (for comparisons)
// --selftest-reservoir  the reservoir unit test (chi-square), then exit
//
// splatting (family C): --accel splat | mlsplat | stochsplat
//                    no photon map at all: the camera pass's visible points go
//                    into a hash grid, filed under every cell their sphere
//                    overlaps, and every photon deposit adds its weighted flux
//                    to the points whose sphere holds it (atomics), in a
//                    photon pass of its own (pipeline __raygen__photon_splat).
//                    Frame: camera -> point grid -> splatting photon pass ->
//                    finalize -> accumulate.  splat: one cell size, F x the
//                    largest radius; mlsplat: each point on the level whose
//                    cells are at least F x its radius; stochsplat: mlsplat's
//                    grid, each deposit adding m times its share to one of the
//                    m points it reaches (reservoir; unbiased).  splat and
//                    mlsplat find exactly the gathers' (photon, point) pairs:
//                    --verify-gather compares every frame with hash / mlhash
//                    over the same gather points and photon paths.  Culling is
//                    implicit (--culling is ignored), the adaptive radius off
//                  --splat-cell-factor F   1 or 2 (default 2)
//                  --splat-stripes K       copies of the accumulators, photon
//                                          i adding to copy i % K (default 1)
//
// optional features (all off by default, see E4):
//                  --inline-positions  hash, mlhash: photon positions next to
//                                     the sorted entries, tested before a
//                                     photon is loaded; splat, mlsplat: the
//                                     points' positions and radii likewise
//                  --hier-mask        culling: a second mask of parent cells,
//                                     tested coarse to fine, so most photons
//                                     are rejected at the coarsest level
//                                     (exact; keeps fewer photons that only a
//                                     mask collision kept)
//                  --fused            family A, offline: after the first frame
//                                     the structure is built before the camera
//                                     pass, from the frame before's radii and
//                                     levels, and the camera walk and lookup
//                                     run as one program (no gather points
//                                     stored); same images
//
// adaptive LoD hash-grid SPPM (family D): --accel sppm_lepski  (also: lepski)
//                    a photon mapper of its own next to the gathers and
//                    splatting, D/sppm_lepski/footprint in logs: its own
//                    OptiX module (sppm_lepski_device.ptx, two pipelines),
//                    kernels and buffers (src/sppm_lepski*).  The other
//                    methods' code is not touched — device.ptx is byte for
//                    byte what it was — and its PTX is only loaded when it
//                    is selected.  A frame is four stages, timed in the
//                    usual fields:
//                      camera  the camera walk — the other photon mappers'
//                              jitter, specular chain and NEE, so the same
//                              vertex and direct light — which also marks
//                              the buckets its gather will search
//                      photon  the photon trace — the other methods' paths —
//                              into 32-byte photons, stored only in marked
//                              buckets and, when weak, by roulette; first
//                              hits are never stored (direct light is NEE)
//                      build   a counting sort into one table of 2^22 buckets
//                      query   the gather at five radii at once (0.25, 0.5,
//                              1, 2, 4 x the base radius; Epanechnikov
//                              kernel), a running mean and variance per
//                              scale (Welford), the scale to show (Lepski's
//                              rule: the largest whose confidence interval
//                              meets all the smaller ones'), and the image
//                    The base radius is one pixel's footprint over the cosine
//                    of incidence (floored at 0.2); --k-pix is the radius of
//                    the largest scale in footprints, so the default 4 makes
//                    the base one pixel.  There is no radius schedule: the
//                    radii stay and the selection moves down the scales.
//                    The grid has one level per octave of distance from the
//                    camera: level l holds the photons d0*2^l .. d0*2^(l+1)
//                    away, in cells h0*2^l wide.  A radius is clamped to
//                    0.49 cells of its level, so a 2x2x2 block per level
//                    holds every photon within it (at most two levels).
//                    Culling is the method's own (--culling is ignored) and,
//                    like the grid, exact: --verify-gather compares every
//                    frame's gather through the grid over the culled map
//                    with a brute-force gather over a map traced without
//                    culling.  --depth up to 64.
//                  --lepski-d0 F      d0 as a fraction of the scene diagonal
//                                     (default 0.01)
//                  --lepski-h0 F      h0 as a fraction of d0 (default 0.02,
//                                     at most 1).  The gather's cost goes
//                                     with the cell's area: at 1280x720 the
//                                     default tests about 250 photons per
//                                     pixel, with radii clamped only at
//                                     grazing angles; 0.08, the value the
//                                     method was specified with, clamps
//                                     nothing and tests about 3800
//                  --lepski-max-lod N levels (default 12)
//                  --lepski-kappa F   half-width of the selection's
//                                     confidence intervals in standard
//                                     errors (default 4).  At 2, the value
//                                     the method was specified with, the
//                                     selection is biased towards dark while
//                                     the small scales hold few photons
//                                     (about 1.5% of the image's brightness
//                                     on the Cornell box at 16 frames, gone
//                                     by 1000)
//                  --lepski-flux-rr F photons below F x the mean emitted
//                                     photon flux are stored with probability
//                                     flux / threshold (at least 0.05) and
//                                     the flux scaled up to match (default
//                                     0.1; 0 = store every one)
//                  --lepski-culling on|off   off stores every photon (the
//                                     same image, to rounding)
//                  --lepski-max-photons N    photon slots (default: paths x
//                                     (depth - 1), at most 16M; a pass that
//                                     outgrows them is reported)
//                  --lepski-view V    normal    direct + indirect at the
//                                               selected scale (default)
//                                     indirect  indirect only
//                                     heatmap   the selected scale: blue,
//                                               cyan, green, yellow, red =
//                                               0.25x .. 4x
//                                     fixed     direct + indirect at 1x, no
//                                               selection (the ablation)
//                    The timing JSON gets a "lepski" record: the settings,
//                    deposits / in active cells / stored per frame (alive
//                    over candidates), the pixels at each scale when the run
//                    ended, the kernels' registers.  --stats works as for
//                    the gathers (photons read, photons within the largest
//                    scale, buckets visited).  The GUI has the method under
//                    Photon map > Acceleration structure, with its view,
//                    kappa and culling.
//
// RTPM (family E): --accel rtpm
//                    the hardware ray tracing photon mapper of Kern, Bruell
//                    and Grosch, "Accelerating Photon Mapping for
//                    Hardware-Based Ray Tracing" (JCGT 12(1), 2023),
//                    E/rtpm/global in logs: its own OptiX module
//                    (rtpm_device.ptx, two pipelines), kernels and buffers
//                    (src/rtpm*).  As with family D nothing of the other
//                    methods is touched, and its PTX is only loaded when it
//                    is selected.  A frame is the renderer's camera pass and
//                    then, timed in the usual fields:
//                      mark    Photon Culling's mask (--rtpm-culling on)
//                      photon  the photon generation — the other methods'
//                              paths — into a caustic map (L S+ D) and a
//                              global map, each photon a box for the
//                              structure and a 32-byte record
//                      build   an acceleration structure per map over this
//                              frame's boxes, rebuilt from scratch
//                      query   the collection: per pixel an infinitely short
//                              ray into each structure, an intersection
//                              program that keeps the photons whose sphere
//                              holds the gather point, an any-hit program
//                              that adds them up
//                    From the paper: the structures and the inverse radius
//                    search, the two maps with their own radii, the rejection
//                    of global photons, Photon Culling, Stochastic
//                    Evaluation.  From this renderer, so that it converges
//                    to the same image and can be checked: the camera pass
//                    and its gather points, direct light by NEE (no
//                    first-hit photons), the shared path-length budget, the
//                    gathers' photon tests and cone filter.  With its
//                    defaults it is therefore an exact structure: the full
//                    hash grid's image (A/hash/global), found another way.
//                    --verify-gather compares every frame's collection
//                    through the structures over the culled maps with a
//                    brute-force one over maps traced without culling.
//                    --radius-mode footprint works too (the boxes are then
//                    as wide as the frame's largest radius).  --culling is
//                    ignored: the method has its own.  --depth up to 64.
//                  --rtpm-culling on|off   Photon Culling (default off): a
//                                     mask of the eight cells nearest to each
//                                     gather point, cells one photon box
//                                     wide; photons elsewhere are not stored
//                  --rtpm-mask-bits N the mask has 2^N cells (default 22, the
//                                     paper's); cells that collide only let
//                                     photons through that nobody collects
//                  --rtpm-reject P    a global photon is stored with
//                                     probability 1 - P and 1 / (1 - P) of
//                                     its flux (default 0; the paper's tests
//                                     0.7); caustic photons always are
//                  --rtpm-stochastic K   Stochastic Evaluation: the any-hit
//                                     program keeps K photon indices by
//                                     reservoir sampling and only those are
//                                     evaluated, each standing for n / K of
//                                     the n found (default 0 = evaluate all;
//                                     the paper's tests 3).  With a diffuse
//                                     BSDF there is no evaluation to save:
//                                     here it costs noise and buys no time
//                  --rtpm-caustic-scale F  the caustic map's radius as a
//                                     fraction of the global one (default 1)
//                  --rtpm-kernel cone|uniform   the gathers' cone filter
//                                     (default), or the paper's estimate,
//                                     every photon in the disc alike
//                  --rtpm-build fast|default|trace   the structures' build
//                                     flags (default fast; no measurable
//                                     difference in build or collection time
//                                     here, fast is a third smaller)
//                  --rtpm-max-photons N   photon slots per map (default: the
//                                     most a pass can deposit, at most 8M)
//                    The paper's test configuration is --rtpm-culling on
//                    --rtpm-reject 0.7 --rtpm-stochastic 3 --rtpm-kernel
//                    uniform.  The timing JSON gets an "rtpm" record: the
//                    settings, deposits / in marked cells / stored per map,
//                    the structures' memory.  --stats: boxes handed to the
//                    intersection program and photons accepted, per pixel.
//                    The GUI has the method under Photon map > Acceleration
//                    structure, with its culling, rejection, list size and
//                    caustic radius.
//
// error against time: --checkpoints [LIST]  at these frame counts (default:
//                                     every power of two up to --frames) save
//                                     the accumulated image as <out>.cpNNNN.exr
//                                     and record the GPU time so far (the sum
//                                     of the pass timers, no file I/O) in
//                                     <out>.checkpoints.json
//                  --dump-frames LIST save single frames (not accumulated) as
//                                     <out>.frame<index>.exr; LIST is 3,8,16 or
//                                     1-64 or 1-1024:16
//   python3 eval/render_references.py   path-traced references (eval/refs)
//   python3 eval/error_vs_time.py       relMSE against GPU time, CSV and SVG
//   python3 eval/variance.py            the fixed-radius variance diagnostic
//   python3 exec.py --experiments [--reduced]   the whole matrix (E1-E9) and
//                                     build/experiments/report.md; E8 is the
//                                     adaptive LoD hash-grid SPPM against the
//                                     hash grids, E9 every method the
//                                     renderer has, path tracing and RTPM
//                                     included (--only E9 for it alone)
//                    configurations are <family>/<accel>/<radius mode> with
//                    variants after a +, one option each (eval/evalio.py):
//                    B/hash/footprint+hier, A/hash/footprint+kpix8,
//                    D/sppm_lepski/footprint+kappa2, E/rtpm/global+cull,
//                    E/rtpm/global+cull+reject0.7+uniform+stoch3,
//                    B/hash/global+hier+paths300000
//   python3 scenes/make_scenes.py       the test scenes stadium, rooms,
//                                     caustic and their camera files
//
// checks of this work (each renders what it needs into build/checks/; the
// "before" build is build/pre_families/build, the sources as they were before
// the method families, see build/pre_families/tree_state.txt):
//   python3 checks.py v1      every existing configuration renders the same
//                             image as before; per-pass times and registers
//   python3 checks.py timing  the per-pass time comparison of v1 on its own
//   python3 checks.py v2      range-based visits: same images, cells and
//                             candidates per lookup before and after, exact
//                             structures against brute force at frames 1,
//                             100, 1000
//   python3 checks.py v3      culling exact per frame, tested entries of
//                             B/mlhash, rect/mlrect unbiased with and without
//                             culling
//   python3 checks.py v4      splat/mlsplat equal hash/mlhash per frame at
//                             frames 1, 100, 1000
//   python3 checks.py v5      rect, mlrect, stochsplat unbiased (1/sqrt(M))
//   python3 checks.py v6      --fused renders the same images as split
//   python3 checks.py v7      error against time: the -2/3 slope (Cornell box)
//   python3 checks.py v8      the adaptive LoD hash-grid SPPM: device.ptx
//                             unchanged (against build/pre_lepski/build), its
//                             grid and culling against brute force on all
//                             four scenes, its brightness against the
//                             path-traced reference
//   python3 checks.py v9      RTPM: device.ptx unchanged, its structures and
//                             culling against brute force on all four
//                             scenes, its image against the full hash
//                             grid's, its rejection and Stochastic
//                             Evaluation against the reference
//
// performance table: python3 exec.py --bench   (build/bench/)
//
// timing a photon-mapped render without scene loading in the number:
//   ./renderer scene.glb --offline --photon --camera camera.txt \
//       --frames 64 --timing-json ../build/timing.json --output ../build/t.exr

// render the whole image set in one call
python3 exec.py                    // path trace + 3 photon renders -> renders/
python3 exec.py --dry-run          // print the render commands only
python3 exec.py --hash-frames 512  // more frames for one of them
//
// the four images:  pt_f1024_d32_s1.exr                       path tracing
//                   pm_hash_f256_d32_p1000000_s1.exr          full hash grid
//                   pm_stochastic_f1024_d32_p1000000_s1.exr   stochastic hash grid
//                   pm_rectified_f1024_d32_p1000000_s1.exr    rectified stochastic
//
// all four share --depth and --seed, the three photon renders also --paths, so
// frame i traces the very same photons in all three.  Frames are per mode
// (--pt-frames, --hash-frames, --stochastic-frames, --rectified-frames): the
// path trace is the reference (and a frame there is 4 camera samples, not 1),
// and the stochastic grids keep one photon per bucket, so their frames are
// cheap but noisy and they get 4x the full hash grid's.  --force re-renders
// images that are already in the output directory; settings and commands land
// in renders/manifest.json
//
// the manifest's "seconds" is whole-process time — scene loading, the
// acceleration build and the PTX compile included.  Each photon render also
// writes <image>.timing.json with the per-pass, per-frame numbers

// code-styling:
// find . -regex '.*\.\(cpp\|hpp\|cu\|cuh\|c\|h\)' -exec clang-format -style=WebKit -i {} \;

// references notes: tinyexr 1.0.2 by syoyo

