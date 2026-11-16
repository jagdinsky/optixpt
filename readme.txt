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
//   python3 exec.py --experiments [--reduced]   the whole matrix (E1-E7) and
//                                     build/experiments/report.md
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

