// Use "P" button for switching between Path Tracing and Photon Mapping

// build directory
cd build

// build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc

// compile
make -j$(nproc)

// run
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
// the one the comparison is about.
//
// gather options:  --accel S          structure the gather searches with:
//                                       grid   uniform grid (default)
//                                       hash   hash grid
//                                       brute  no structure; test every photon
//                  --brute-force      shorthand for --accel brute
//                  --verify-gather    run the query pass twice per frame, the
//                                     selected structure against brute force,
//                                     and report how far apart the two images
//                                     are (exits non-zero if they disagree)
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
// timing JSON, per frame: radius, photons, cells (uniform grid only), buckets
// (hash grid only), and photon_trace_ms, build_ms, camera_ms, query_ms,
// accum_ms
//
// python3 exec.py --accel hash       // the photon image set with the hash grid;
//                                    // file names get a _hash suffix
//
// timing a photon-mapped render without scene loading in the number:
//   ./renderer scene.glb --offline --photon --camera camera.txt \
//       --frames 64 --timing-json ../build/timing.json --output ../build/t.exr

// render the whole image set in one call
python3 exec.py                    // reference path trace + 4 photon renders -> renders/
python3 exec.py --dry-run          // print the render commands only
python3 exec.py --budget 600       // give the reference 10 minutes instead of 4
//
// the reference is a depth-32 path trace with as many frames as fit in --budget
// (frame count fitted from two short probe renders); the photon images are the
// four corners of 32/128 frames x depth 8/32.  --force re-renders images that
// are already in the output directory; timings land in renders/manifest.json
//
// the manifest's "seconds" is whole-process time — scene loading, the
// acceleration build and the PTX compile included.  Each photon render also
// writes <image>.timing.json with the per-pass, per-frame numbers, and the
// manifest carries their warm-up-free means under "timing"

// code-styling:
// find . -regex '.*\.\(cpp\|hpp\|cu\|cuh\|c\|h\)' -exec clang-format -style=WebKit -i {} \;

// references notes: tinyexr 1.0.2 by syoyo

