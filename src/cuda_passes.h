#pragma once
#include "params.h"
#include <cuda_runtime.h>

// The parts of a frame that run as plain CUDA kernels rather than OptiX
// launches: the photon-grid build and the accumulation of a finished frame.
//
// They live in their own translation unit because they are compiled by nvcc
// into the executable (device.cu, by contrast, is compiled to PTX and handed to
// OptiX).  That is also what makes CUB available, which is where the sort and
// the prefix sum behind the grid build come from.

// What the build actually produced.  The cell size is reported because it is
// not always the gather radius: the uniform grid's cell budget can force it
// up, and a grid whose cells are wider than r is a different measurement.
struct GridBuildStats {
    long long cells = 0; // uniform grid: cells in the dense array (0 for the hash grid)
    long long buckets = 0; // hash grid: buckets in the table (0 for the uniform grid)
    int3 dims = { 0, 0, 0 };
    float cell_size = 0.f;
    float radius = 0.f;
    int photons = 0;
    bool cell_size_capped = false;
};

// Device scratch for the build, owned by the caller and shared by both grids.
// Buffers are grown when they no longer fit and otherwise reused, so a
// steady-state frame allocates nothing.  `cell_start` and `ids_out` are the two
// the gather reads; the rest is working memory.
struct GridScratch {
    int* cell_start = nullptr; // [cells or buckets + 1] offsets -> PhotonGrid::cell_start
    size_t cell_start_cap = 0; // in bytes
    int* cell_counts = nullptr; // [cells or buckets + 1] histogram, scanned into cell_start
    size_t cell_counts_cap = 0; // in bytes
    int* keys_in = nullptr; // cell index per photon
    int* keys_out = nullptr; // the same, sorted
    int* ids_in = nullptr; // 0..n-1
    int* ids_out = nullptr; // photon ids sorted by key -> grid_photon_ids
    size_t photon_cap = 0; // in elements
    void* temp = nullptr; // CUB working storage
    size_t temp_cap = 0; // in bytes
};

// Bins `storedCount` photons into the uniform grid and points params.grid at
// the result.  Everything stays on the device; the photon map is never read
// back.  Sets params.accel.
void buildPhotonGridGPU(Params& params, int storedCount, GridScratch& scratch,
    cudaStream_t stream, GridBuildStats* stats);

// The same, into the hash grid.  The table has a power-of-two number of
// buckets, at least twice the number of photons, so its size depends on the
// photon count and not on the radius.  Sets params.accel.
void buildPhotonHashGridGPU(Params& params, int storedCount, GridScratch& scratch,
    cudaStream_t stream, GridBuildStats* stats);

void freeGridScratch(GridScratch& scratch);

// accum += query_radiance, then tone-map the running mean into frame_buffer.
// Kept out of the query pass so that the pass being benchmarked does nothing
// but look photons up.
void launchAccumulate(const Params& params, cudaStream_t stream);
