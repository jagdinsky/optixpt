#include "cuda_passes.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_PASS_CHECK(call)                                                 \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[cuda_passes] %s:%d %s -> %s\n", __FILE__,  \
                __LINE__, #call, cudaGetErrorString(_e));                     \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

// The dense part of the grid is one int per cell, and the cell count grows as
// r^-3 while the PPM schedule shrinks r — so on a card with a few GB the grid
// runs out of memory long before the photon map does.  The budget below is the
// point where the build stops subdividing and starts growing the cell instead.
//
// Growing the cell is the safe direction: the gather derives its cell index
// from grid.cell_size and walks 3x3x3 cells, which covers the search sphere for
// any cell_size >= r.  Clamping the dimensions while leaving the cell at r
// would not — the two ends would disagree about which cell a point is in, and
// the gather would silently lose photons.
static constexpr long long kMaxGridCells = 32LL << 20; // 32M cells = 128 MB

// The hash table has at least twice as many buckets as there are photons.
// There are never more occupied cells than photons, so this keeps the average
// number of occupied cells per bucket (the load factor) at or below 0.5 — and
// in practice far lower, because a cell holds many photons.  The cap keeps a
// full 32M-photon map from asking for more than 64M buckets (256 MB).
static constexpr long long kMinHashBuckets = 1LL << 10;
static constexpr long long kMaxHashBuckets = 1LL << 26;

static constexpr int kBlock = 256;

static inline int blocksFor(int n) { return (n + kBlock - 1) / kBlock; }

// Uniform grid key: the cell's position in the dense array.  Must agree exactly
// with gatherPhotonsGrid in device.cu; photonCellCoord is shared for that
// reason.
__global__ void photonCellKeysKernel(const Photon* __restrict__ photons, int n,
    float3 bmin, float cellSize, int3 dims,
    int* __restrict__ keys, int* __restrict__ ids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float3 p = photons[i].pos;
    int ix = photonCellCoord(p.x, bmin.x, cellSize);
    int iy = photonCellCoord(p.y, bmin.y, cellSize);
    int iz = photonCellCoord(p.z, bmin.z, cellSize);
    ix = max(0, min(ix, dims.x - 1));
    iy = max(0, min(iy, dims.y - 1));
    iz = max(0, min(iz, dims.z - 1));

    keys[i] = (iz * dims.y + iy) * dims.x + ix;
    ids[i] = i;
}

// Hash grid key: the bucket the photon's cell hashes to.  Must agree exactly
// with gatherPhotonsHash in device.cu; photonCellCoord and hashCell are shared
// for that reason.  No clamping — every integer cell has a bucket.
__global__ void photonHashKeysKernel(const Photon* __restrict__ photons, int n,
    float3 origin, float cellSize, unsigned int hashMask,
    int* __restrict__ keys, int* __restrict__ ids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float3 p = photons[i].pos;
    const int ix = photonCellCoord(p.x, origin.x, cellSize);
    const int iy = photonCellCoord(p.y, origin.y, cellSize);
    const int iz = photonCellCoord(p.z, origin.z, cellSize);

    keys[i] = (int)(hashCell(ix, iy, iz) & hashMask);
    ids[i] = i;
}

__global__ void countCellsKernel(const int* __restrict__ keys, int n,
    int* __restrict__ counts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    atomicAdd(&counts[keys[i]], 1);
}

__global__ void accumulateKernel(float3* __restrict__ accum,
    uchar4* __restrict__ frame,
    const float3* __restrict__ radiance,
    int n, int frameIndex)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float3 r = radiance[i];
    const float3 prev = (frameIndex == 0) ? make_float3(0.f, 0.f, 0.f) : accum[i];
    const float3 acc = make_float3(prev.x + r.x, prev.y + r.y, prev.z + r.z);
    accum[i] = acc;

    if (!frame)
        return;

    // Running mean, clamped, gamma 2.0 — the same tone map the path tracer
    // applies at the end of __raygen__rg.
    const float nf = float(frameIndex + 1);
    const float mr = sqrtf(fminf(fmaxf(acc.x / nf, 0.f), 1.f));
    const float mg = sqrtf(fminf(fmaxf(acc.y / nf, 0.f), 1.f));
    const float mb = sqrtf(fminf(fmaxf(acc.z / nf, 0.f), 1.f));
    frame[i] = make_uchar4((unsigned char)(mr * 255.f),
        (unsigned char)(mg * 255.f),
        (unsigned char)(mb * 255.f), 255u);
}

static void ensureCapacity(void** ptr, size_t& capacity, size_t needed)
{
    if (*ptr && capacity >= needed)
        return;
    if (*ptr)
        CUDA_PASS_CHECK(cudaFree(*ptr));
    capacity = needed + needed / 4; // a little headroom, as on the host side
    CUDA_PASS_CHECK(cudaMalloc(ptr, capacity));
}

static long long dimsForCellSize(const float ext[3], float cellSize, int3& dims)
{
    dims.x = std::max(1, (int)std::ceil(ext[0] / cellSize));
    dims.y = std::max(1, (int)std::ceil(ext[1] / cellSize));
    dims.z = std::max(1, (int)std::ceil(ext[2] / cellSize));
    return (long long)dims.x * (long long)dims.y * (long long)dims.z;
}

// Makes every build buffer large enough for `storedCount` photons and
// `numKeys` distinct key values (cells for the uniform grid, buckets for the
// hash grid).  Only allocates when something has outgrown its buffer.
static void reserveBuildBuffers(GridScratch& s, int storedCount, long long numKeys)
{
    const size_t numOffsets = (size_t)numKeys + 1;
    ensureCapacity((void**)&s.cell_start, s.cell_start_cap, numOffsets * sizeof(int));
    ensureCapacity((void**)&s.cell_counts, s.cell_counts_cap, numOffsets * sizeof(int));
    if (!s.keys_in || s.photon_cap < (size_t)storedCount) {
        if (s.keys_in) {
            CUDA_PASS_CHECK(cudaFree(s.keys_in));
            CUDA_PASS_CHECK(cudaFree(s.keys_out));
            CUDA_PASS_CHECK(cudaFree(s.ids_in));
            CUDA_PASS_CHECK(cudaFree(s.ids_out));
        }
        s.photon_cap = (size_t)storedCount + (size_t)storedCount / 4;
        const size_t bytes = s.photon_cap * sizeof(int);
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.keys_in, bytes));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.keys_out, bytes));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.ids_in, bytes));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.ids_out, bytes));
    }
}

// The part of the build both grids share, run once keys_in and ids_in have
// been written: group the photon ids by key.
//
// Keys are integers in [0, numKeys).  A histogram of them followed by an
// exclusive prefix sum gives each key's starting offset; a stable radix sort of
// the (key, id) pairs then lays the ids down in exactly that order.
static void groupPhotonsByKey(GridScratch& s, int storedCount, long long numKeys,
    cudaStream_t stream)
{
    const int numOffsets = (int)(numKeys + 1);

    // Only the bits the key actually uses need sorting.
    int endBit = 1;
    while ((1LL << endBit) < numKeys)
        ++endBit;

    size_t scanBytes = 0;
    CUDA_PASS_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes,
        s.cell_counts, s.cell_start, numOffsets, stream));
    size_t sortBytes = 0;
    CUDA_PASS_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes,
        s.keys_in, s.keys_out, s.ids_in, s.ids_out, storedCount, 0, endBit, stream));
    ensureCapacity(&s.temp, s.temp_cap, std::max(scanBytes, sortBytes));

    // Histogram of keys -> exclusive prefix sum = offsets.  numOffsets entries,
    // the last count staying zero, so cell_start[numKeys] ends up holding the
    // total and every entry can be read as a [start, end) pair.
    CUDA_PASS_CHECK(cudaMemsetAsync(s.cell_counts, 0,
        (size_t)numOffsets * sizeof(int), stream));
    countCellsKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
        s.keys_in, storedCount, s.cell_counts);
    CUDA_PASS_CHECK(cub::DeviceScan::ExclusiveSum(s.temp, scanBytes,
        s.cell_counts, s.cell_start, numOffsets, stream));

    // Photon ids grouped by key.  The radix sort is stable and the ids go in
    // ascending, so each key lists its photons in photon-map order — the same
    // order every run, which is what makes two renders bit-for-bit comparable.
    CUDA_PASS_CHECK(cub::DeviceRadixSort::SortPairs(s.temp, sortBytes,
        s.keys_in, s.keys_out, s.ids_in, s.ids_out, storedCount, 0, endBit, stream));
}

void buildPhotonGridGPU(Params& params, int storedCount, GridScratch& s,
    cudaStream_t stream, GridBuildStats* stats)
{
    if (stats)
        *stats = GridBuildStats();

    storedCount = std::min(storedCount, params.photon_capacity);
    if (storedCount <= 0) {
        params.accel = ACCEL_BRUTE_FORCE;
        return;
    }

    // Bounds: the scene box padded by one radius, not a reduction over the
    // photons.  Photons only land on geometry, so the scene box contains them,
    // and taking it from the host means the build never has to read anything
    // back from the device.  The padding keeps a photon on the boundary inside
    // the grid the way the old per-photon box did.
    const float r = params.gather_radius;
    const float3 bmin = make_float3(params.scene_aabb_min.x - r,
        params.scene_aabb_min.y - r,
        params.scene_aabb_min.z - r);
    const float3 bmax = make_float3(params.scene_aabb_max.x + r,
        params.scene_aabb_max.y + r,
        params.scene_aabb_max.z + r);
    const float ext[3] = { bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z };

    float cellSize = r;
    int3 dims;
    long long cells = dimsForCellSize(ext, cellSize, dims);
    bool capped = false;
    for (int guard = 0; cells > kMaxGridCells && guard < 32; ++guard) {
        capped = true;
        const double grow = std::cbrt((double)cells / (double)kMaxGridCells);
        cellSize *= (float)std::max(grow, 1.01);
        cells = dimsForCellSize(ext, cellSize, dims);
    }

    reserveBuildBuffers(s, storedCount, cells);

    // 1. one cell index per photon
    photonCellKeysKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
        params.photon_map, storedCount, bmin, cellSize, dims, s.keys_in, s.ids_in);

    // 2 and 3. histogram -> prefix sum -> sort
    groupPhotonsByKey(s, storedCount, cells, stream);

    params.grid.aabb_min = bmin;
    params.grid.aabb_max = bmax;
    params.grid.dims = dims;
    params.grid.cell_size = cellSize;
    params.grid.hash_mask = 0;
    params.grid.cell_start = s.cell_start;
    params.grid.grid_photon_ids = s.ids_out;
    params.accel = ACCEL_UNIFORM_GRID;

    if (stats) {
        stats->cells = cells;
        stats->dims = dims;
        stats->cell_size = cellSize;
        stats->radius = r;
        stats->photons = storedCount;
        stats->cell_size_capped = capped;
    }
}

void buildPhotonHashGridGPU(Params& params, int storedCount, GridScratch& s,
    cudaStream_t stream, GridBuildStats* stats)
{
    if (stats)
        *stats = GridBuildStats();

    storedCount = std::min(storedCount, params.photon_capacity);
    if (storedCount <= 0) {
        params.accel = ACCEL_BRUTE_FORCE;
        return;
    }

    // The cell size is always exactly r.  The hash grid has no dense array, so
    // there is no memory budget that could force the cell to grow.  The scene
    // box is used only as the origin of the cell coordinates (padded by r, as
    // for the uniform grid, so coordinates of photons are never negative);
    // there is no upper bound.
    const float r = params.gather_radius;
    const float3 origin = make_float3(params.scene_aabb_min.x - r,
        params.scene_aabb_min.y - r,
        params.scene_aabb_min.z - r);

    // Number of buckets: the smallest power of two >= 2 * photons, within the
    // limits above.  A power of two turns "hash mod buckets" into "hash & mask".
    long long buckets = kMinHashBuckets;
    while (buckets < 2LL * storedCount && buckets < kMaxHashBuckets)
        buckets <<= 1;
    const unsigned int mask = (unsigned int)(buckets - 1);

    reserveBuildBuffers(s, storedCount, buckets);

    // 1. one bucket index per photon
    photonHashKeysKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
        params.photon_map, storedCount, origin, r, mask, s.keys_in, s.ids_in);

    // 2 and 3. histogram -> prefix sum -> sort, exactly as for the uniform grid
    groupPhotonsByKey(s, storedCount, buckets, stream);

    params.grid.aabb_min = origin;
    params.grid.aabb_max = make_float3(params.scene_aabb_max.x + r,
        params.scene_aabb_max.y + r,
        params.scene_aabb_max.z + r);
    params.grid.dims = make_int3(0, 0, 0);
    params.grid.cell_size = r;
    params.grid.hash_mask = mask;
    params.grid.cell_start = s.cell_start;
    params.grid.grid_photon_ids = s.ids_out;
    params.accel = ACCEL_HASH_GRID;

    if (stats) {
        stats->buckets = buckets;
        stats->cell_size = r;
        stats->radius = r;
        stats->photons = storedCount;
    }
}

void freeGridScratch(GridScratch& s)
{
    auto release = [](void* p) {
        if (p)
            cudaFree(p);
    };
    release(s.cell_start);
    release(s.cell_counts);
    release(s.keys_in);
    release(s.keys_out);
    release(s.ids_in);
    release(s.ids_out);
    release(s.temp);
    s = GridScratch();
}

void launchAccumulate(const Params& params, cudaStream_t stream)
{
    const int n = (int)(params.width * params.height);
    accumulateKernel<<<blocksFor(n), kBlock, 0, stream>>>(
        params.accum_buffer, params.frame_buffer, params.query_radiance,
        n, params.frame_index);
    CUDA_PASS_CHECK(cudaGetLastError());
}
