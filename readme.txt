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

// render the whole image set in one call
python3 exec.py                    // reference path trace + 4 photon renders -> renders/
python3 exec.py --dry-run          // print the render commands only
python3 exec.py --budget 600       // give the reference 10 minutes instead of 4
//
// the reference is a depth-32 path trace with as many frames as fit in --budget
// (frame count fitted from two short probe renders); the photon images are the
// four corners of 32/128 frames x depth 8/32.  --force re-renders images that
// are already in the output directory; timings land in renders/manifest.json

// code-styling:
// find . -regex '.*\.\(cpp\|hpp\|cu\|cuh\|c\|h\)' -exec clang-format -style=WebKit -i {} \;

// references notes: tinyexr 1.0.2 by syoyo

