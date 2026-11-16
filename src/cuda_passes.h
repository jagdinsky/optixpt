#pragma once
#include "params.h"
#include <cuda_runtime.h>
#include <vector>

// The parts of a frame that run as plain CUDA kernels rather than OptiX
// launches: the photon-grid build, the photon lookup (query pass), the
// culling marks and the accumulation of a finished frame.
//
// They live in their own translation unit because they are compiled by nvcc
// into the executable (device.cu, by contrast, is compiled to PTX and handed to
// OptiX).  That is also what makes CUB available, which is where the sort and
// the prefix sum behind the grid build come from.  The lookup is pure compute
// — no ray is traced — so it needs nothing from OptiX, and as a kernel per
// structure it gets its own register allocation; see gather.h.

// What the build actually produced.  The cell size is reported because it is
// not always the gather radius: the uniform grid's cell budget can force it
// up, and a grid whose cells are wider than r is a different measurement.
struct GridBuildStats {
    long long cells = 0; // uniform grid: cells in the dense array (0 for the hash grids)
    long long buckets = 0; // hash grids: buckets in the table (0 for the uniform grid)
    long long entries = 0; // multi-level grids: (photon, level) pairs filed
    long long bytes = 0; // memory the lookup reads: the structure, not the photons
    int3 dims = { 0, 0, 0 };
    float cell_size = 0.f;
    float radius = 0.f;
    int photons = 0;
    bool cell_size_capped = false;
};

// Device scratch for the build, owned by the caller and shared by all grids.
// Buffers are grown when they no longer fit and otherwise reused, so a
// steady-state frame allocates nothing.  `cell_start` and `ids_out` are the two
// the uniform and full hash grid gathers read; the bucket_* tables are the
// stochastic hash grids' whole structure; the rest is working memory.
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

    int* bucket_count = nullptr; // -> PhotonGrid::bucket_count
    size_t bucket_count_cap = 0; // in bytes
    Photon* bucket_photon = nullptr; // -> PhotonGrid::bucket_photon (stochastic)
    size_t bucket_photon_cap = 0; // in bytes
    unsigned long long* bucket_pick = nullptr; // -> PhotonGrid::bucket_pick (rectified)
    size_t bucket_pick_cap = 0; // in bytes

    void* radius_stats = nullptr; // reduceGatherRadii's one-element result
    size_t radius_stats_cap = 0; // in bytes

    // Multi-level grids with culling: every (photon, level) candidate before
    // the ones in cells no lookup marked are compacted away.
    unsigned int* ml_keys = nullptr; // full hashCell4 of the entry's cell
    int* ml_values = nullptr; // photon index | level << LEVEL_SHIFT
    unsigned char* ml_flags = nullptr; // 1 = the cell is marked
    size_t ml_cap = 0; // in entries
    int* ml_selected = nullptr; // entries that survived the compaction

    // --stats summaries
    float4* positions = nullptr; // --inline-positions: -> PhotonGrid::positions
    size_t positions_cap = 0; // in elements

    int* stat_sorted = nullptr; // a sorted copy of one per-pixel counter
    size_t stat_sorted_cap = 0; // in bytes
    long long* stat_scalars = nullptr; // [2] lookups, sum

    // Splatting: the visible-point grid's build.  cell_start / cell_counts
    // above are its offsets too (no photon grid is built in that family).
    int* vp_counts = nullptr; // [points + 1] cells per visible point, then offsets
    int* vp_offsets = nullptr; // [points + 1]
    size_t vp_points_cap = 0; // in elements
    unsigned long long* vp_keys_in = nullptr; // bucket << 32 | entry, per (point, cell)
    unsigned long long* vp_keys_out = nullptr; // the same, sorted by bucket
    unsigned int* vp_values = nullptr; // -> VisiblePointGrid::values
    float4* vp_points = nullptr; // --inline-positions: -> VisiblePointGrid::points
    size_t vp_entries_cap = 0; // in elements
    int* vp_scalars = nullptr; // [2 + MAX_LEVELS] total entries, unique entries, level flags
};

// What the visible-point grid build produced.
struct VpGridStats {
    long long entries = 0; // (point, cell) pairs filed
    long long unique = 0; // after removing a point's second entry in a bucket
    long long buckets = 0;
    long long bytes = 0; // what the splatting pass reads: offsets and entries
    int num_levels = 0;
    int level_min = -1;
    int level_max = -1;
};

// Bins `storedCount` photons into the uniform grid and points params.grid at
// the result.  `radius` is the largest radius any lookup of the frame will
// use (gather_radius, or the largest per-pixel radius): the cells are at least
// that wide, which is what lets a lookup of any smaller radius walk the same
// 3x3x3 cells.  Everything stays on the device; the photon map is never read
// back.  Sets params.accel.
void buildPhotonGridGPU(Params& params, int storedCount, float radius,
    GridScratch& scratch, cudaStream_t stream, GridBuildStats* stats);

// The same, into the hash grid.  The table has a power-of-two number of
// buckets, at least params.table_load per photon, so its size depends on the
// photon count and not on the radius.  Sets params.accel.
void buildPhotonHashGridGPU(Params& params, int storedCount, float radius,
    GridScratch& scratch, cudaStream_t stream, GridBuildStats* stats);

// The stochastic hash grids (`type` is ACCEL_STOCHASTIC_HASH_GRID or
// ACCEL_RECTIFIED_HASH_GRID) are not built from the photon map: the photon
// pass files each photon into them as it deposits it.  So this runs *before*
// the trace, and all it does is size the table, clear it, and point
// params.grid at it.  Same cells and hash as the full hash grid.  Sets
// params.accel.
void prepareStochasticHashGridGPU(Params& params, AccelType type, float radius,
    GridScratch& scratch, cudaStream_t stream, GridBuildStats* stats);

// The multi-level hash grid: every stored photon is filed once per level in
// params.levels, under its own cell on that level, so each lookup can search
// the level that fits its radius — cells between r and 2r wide whatever r
// is.  With culling, a (photon, level) entry whose cell no lookup marked is
// left out (no lookup on that level searches it).  The table has a power of
// two buckets, at least params.table_load per entry.  Sets params.accel.
void buildMultiLevelHashGridGPU(Params& params, int storedCount, GridScratch& scratch,
    cudaStream_t stream, GridBuildStats* stats);

// The multi-level rectified grid: the same insertions as the multi-level hash
// grid, but each bucket keeps only a count and one representative, picked
// uniformly among everything inserted there by the smallest of independent
// random keys (see insertRepresentative).  `bucketsOverride` > 0 fixes the
// table size (a power of two); otherwise it has params.table_load buckets per
// insertion, which with culling are counted first.  Built after the photon
// pass, like the full grids.  Sets params.accel.
void buildMultiLevelRectGridGPU(Params& params, int storedCount, long long bucketsOverride,
    GridScratch& scratch, cudaStream_t stream, GridBuildStats* stats);

// --mlrect-in-trace: the photon pass inserts into the multi-level rectified
// grid as it stores each photon (insertRepresentative, the same insertions
// and keys as the kernel), so this runs *before* the trace and only sizes the
// table — from the photon capacity, the stored count not being known yet —
// clears it and points params.grid at it.  Sets params.accel.
void prepareMultiLevelRectGridForTraceGPU(Params& params, long long bucketsOverride,
    GridScratch& scratch, cudaStream_t stream, GridBuildStats* stats);

// How many insertions went into the multi-level rectified grid (the sum of
// its bucket counts).  Statistics only, for when culling makes the number
// unknown in advance.
long long countMultiLevelRectEntries(const Params& params, GridScratch& scratch,
    cudaStream_t stream);

// The reservoir unit test: `reps` buckets with `n` insertions each, every
// repetition with its own seed, run twice — the insertions of a bucket in
// neighbouring threads, and scattered over the whole launch.  Prints the
// chi-square statistic of how often each insertion won; false if either run
// fails at the 1% level or a bucket's count is off.
bool runReservoirSelfTest(int n, int reps);

// Mean, 95th percentile (nearest rank) and maximum of a per-pixel counter
// written by the query pass under --stats, over the pixels that have a lookup
// (the others hold -1).  Sorted and reduced on the device; four scalars come
// back.
struct QueryStatSummary {
    double mean = 0.0;
    int p95 = 0;
    int max = 0;
    long long lookups = 0;
};
void summarizeQueryStats(const int* perPixel, int pixels, GridScratch& scratch,
    cudaStream_t stream, QueryStatSummary& out);

// What the frame's lookups asked for, over the valid gather points.
struct GatherRadiusStats {
    float max_r = 0.f;
    double sum_r = 0.0;
    double sum_px = 0.0; // r / (t * pixel_angle): the radius in pixel footprints
    long long valid = 0;
};

// Reduces over params.gather_points on the device and reads back only the
// result.  The maximum is what a single-level grid has to size its cells by
// when every pixel has its own radius.
void reduceGatherRadii(const Params& params, GridScratch& scratch,
    cudaStream_t stream, GatherRadiusStats& out);

void freeGridScratch(GridScratch& scratch);

// accum += query_radiance, then tone-map the running mean into frame_buffer.
// Kept out of the query pass so that the pass being benchmarked does nothing
// but look photons up.
void launchAccumulate(const Params& params, cudaStream_t stream);

// The query pass: every pixel's photon lookup through the structure
// params.accel names, into query_radiance, and with params.stats into the
// per-pixel counters too.  One kernel per structure and per --stats setting.
void launchQueryKernel(const Params& params, cudaStream_t stream);

// Photon culling, the lookup side: sets in params.cull_mask, which the caller
// has cleared, the bit of every cell a valid gather point's sphere overlaps at
// its own level.
void launchMarkKernel(const Params& params, cudaStream_t stream);

// Splatting: files every valid gather point under every cell its sphere
// overlaps on its level — one cell size params.vpgrid.cell_size (> 0, set by
// the caller: splat) or level levelOf(F x r) (cell_size = 0: mlsplat) — in a
// hash table of params.table_load buckets per entry, leaving each point at
// most once per bucket.  Sets params.vpgrid.  Reads back three scalars (the
// entries before and after the de-duplication) and the 32 level flags.
void buildVisiblePointGridGPU(Params& params, GridScratch& scratch, cudaStream_t stream,
    VpGridStats* stats);

// Splatting: query_radiance = direct + weight * 3 / (pi r^2) * flux, the flux
// summed over the accumulator copies; the same formula the gathers end with.
void launchSplatFinalize(const Params& params, cudaStream_t stream);

// What the compiler gave each kernel whose time is compared: registers per
// thread and local memory (spills and local arrays) per thread.
struct KernelResources {
    const char* name;
    int registers;
    size_t local_bytes;
    int max_threads_per_block;
};
std::vector<KernelResources> kernelResources();
