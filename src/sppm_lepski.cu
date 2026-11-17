// The CUDA side of the adaptive LoD hash-grid SPPM (sppm_lepski.h): its
// device memory, stage 3 (the counting sort of the photons into the hash
// table) and stage 4 (the multi-scale gather, the running estimate, the
// choice of scale and the image).
//
// A translation unit of its own, like cuda_passes.cu, compiled by nvcc into
// the executable; nothing here is called by the other methods.
#include "sppm_lepski.h"

#include <cub/cub.cuh>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

#define LEPSKI_CHECK(call)                                                   \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::fprintf(stderr, "[sppm_lepski] %s:%d %s -> %s\n", __FILE__, \
                __LINE__, #call, cudaGetErrorString(_e));                    \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

static constexpr int kLepskiBlock = 256;
static inline int lepskiBlocksFor(size_t n) { return (int)((n + kLepskiBlock - 1) / kLepskiBlock); }

// The table has one entry more than there are buckets, and its last count
// stays zero: the scan then leaves the total there, and bucket b's photons
// are sorted_photons[cell_start[b] .. cell_start[b + 1]) — one array to read
// in the gather rather than a start and a count.
static constexpr size_t kLepskiTableEntries = (size_t)LEPSKI_HASH_SIZE + 1;
static constexpr size_t kLepskiBitsetBytes = (size_t)(LEPSKI_HASH_SIZE / 32) * sizeof(unsigned int);

// --- Stage 3: counting sort -----------------------------------------------

__global__ void lepskiCountKernel(const unsigned int* __restrict__ hashes, int n,
    unsigned int* __restrict__ counts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    atomicAdd(&counts[hashes[i]], 1u);
}

// Every photon takes the next free place of its bucket.  The order within a
// bucket is whatever order the threads arrive in; the gather sums a bucket,
// so only the rounding of that sum depends on it.
__global__ void lepskiScatterKernel(const unsigned int* __restrict__ hashes,
    const PackedPhoton* __restrict__ raw, int n, unsigned int* __restrict__ cursor,
    PackedPhoton* __restrict__ sorted)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const unsigned int dst = atomicAdd(&cursor[hashes[i]], 1u);
    sorted[dst] = raw[i];
}

// --- Stage 4: gather --------------------------------------------------------

// One gather: what it compares a photon against, and what it has summed.
struct LepskiLookup {
    float3 x;
    float3 n;
    int maxPhotonDepth;
    float rMax2; // the largest scale's radius, squared
    float planeTol;
    float invR2[LEPSKI_K_SCALES];
    float3 flux[LEPSKI_K_SCALES]; // kernel-weighted flux at every scale
    unsigned int accepted; // photons within the largest scale
    int tested; // photon records read
    int cells; // buckets visited
};

__device__ __forceinline__ void lepskiAccumulate(LepskiLookup& q, const PackedPhoton& ph)
{
    const float3 diff = ph.pos - q.x;
    const float d2 = dot(diff, diff);
    if (d2 >= q.rMax2)
        return;
    const unsigned int nd = ph.nOct;
    // the path-length budget shared with the camera walk
    if (unpackDepth(nd) > q.maxPhotonDepth)
        return;
    // on the surface being shaded: a matching normal, and within a thin slab
    // around the tangent plane
    if (dot(q.n, unpackNormal(nd)) < LEPSKI_NORMAL_TOL)
        return;
    if (fabsf(dot(diff, q.n)) > q.planeTol)
        return;
    // The BSDF without the cosine: a Lambertian surface reflects albedo / pi
    // of what arrives from in front of it (the factor is in the gather
    // point's weight) and nothing of what arrives from behind.
    if (dot(q.n, UnpackOctahedral(ph.wiOct)) <= 0.f)
        return;

    ++q.accepted;

    // The 2D Epanechnikov kernel, 2 / (pi r^2) (1 - d^2 / r^2), which
    // integrates to one over the disc of radius r — at all the scales at
    // once, in registers, with no square root.
    const float3 flux = ph.flux;
#pragma unroll
    for (int k = 0; k < LEPSKI_K_SCALES; ++k) {
        const float u2 = d2 * q.invR2[k];
        if (u2 < 1.f)
            q.flux[k] += ((2.f / RT_PI * q.invR2[k]) * (1.f - u2)) * flux;
    }
}

// One pixel's gather of this iteration; false if the pixel gathers nothing.
// BRUTE tests every stored photon instead of the buckets around the gather
// point: the reference the grid is checked against (--verify-gather).
template <bool BRUTE>
__device__ __forceinline__ bool lepskiGatherPixel(const LepskiParams& L,
    const LepskiGatherPoint& gp, LepskiLookup& q)
{
#pragma unroll
    for (int k = 0; k < LEPSKI_K_SCALES; ++k)
        q.flux[k] = make_float3(0.f, 0.f, 0.f);
    q.accepted = 0u;
    q.tested = 0;
    q.cells = 0;

    const float rBase = gp.r_base;
    if (!(rBase > 0.f))
        return false;

    const float rMax = lepskiScale(LEPSKI_K_SCALES - 1) * rBase;
    q.x = gp.pos;
    q.n = gp.normal;
    q.maxPhotonDepth = lepskiMaxPhotonDepth(gp.info);
    q.rMax2 = rMax * rMax;
    q.planeTol = LEPSKI_PLANE_TOL * rMax;
#pragma unroll
    for (int k = 0; k < LEPSKI_K_SCALES; ++k) {
        const float r = lepskiScale(k) * rBase;
        q.invR2[k] = 1.f / (r * r);
    }

    if (BRUTE) {
        const unsigned int stored = L.cell_start[LEPSKI_HASH_SIZE];
        q.tested = (int)stored;
        for (unsigned int e = 0; e < stored; ++e)
            lepskiAccumulate(q, L.sorted_photons[e]);
    } else {
        unsigned int buckets[16];
        q.cells = lepskiGatherBuckets(L, q.x, lepskiLevelMin(gp.info), lepskiLevelMax(gp.info),
            buckets);
        for (int b = 0; b < q.cells; ++b) {
            const unsigned int begin = L.cell_start[buckets[b]];
            const unsigned int end = L.cell_start[buckets[b] + 1];
            q.tested += (int)(end - begin);
            for (unsigned int e = begin; e < end; ++e)
                lepskiAccumulate(q, L.sorted_photons[e]);
        }
    }
    return true;
}

// The scale a pixel's running estimate is shown at: Lepski's rule in its
// intersection-of-confidence-intervals form.
//
// Every scale estimates the same pixel with its own bias and noise: a small
// radius is nearly unbiased and noisy, a large one smooth and — where the
// light changes within it — biased.  Each has a confidence interval around
// its running mean, kappa standard errors either way.  Going from the
// smallest scale up, the intervals are intersected, and the estimate is taken
// at the largest scale whose interval still meets all the smaller ones': up
// to there the scales agree within their noise, and the largest of them is
// the least noisy; past it the bias of the larger scale has shown.
//
// The standard error is the larger of two.  One is measured: the Welford
// variance of the iterations' estimates, over M.  The other is a floor, one
// photon's worth: the mean of the largest scale divided by the photons that
// made it, scaled to the kernel of scale k (whose peak is that much higher).
// A small scale that has not caught a single photon yet has measured variance
// zero around a mean of zero — an interval of no width at a wrong value —
// and the floor is what keeps it from vetoing every scale above it.
__device__ __forceinline__ int lepskiSelectScale(const PixelLepskiState& st, int M, float kappa)
{
    // No variance to speak of before the third iteration: scale 1.
    if (M < 3)
        return 2;

    const float invMM = 1.f / ((float)M * (float)(M - 1));
    const float perPhoton = st.meanLum[LEPSKI_K_SCALES - 1]
        / (float)max(st.totalCountMax, 1u);
    const float maxScale = lepskiScale(LEPSKI_K_SCALES - 1);

    int best = 0;
    float lowMax = -FLT_MAX;
    float highMin = FLT_MAX;
#pragma unroll
    for (int k = 0; k < LEPSKI_K_SCALES; ++k) {
        const float sigmaWelford = sqrtf(fmaxf(0.f, st.m2Lum[k] * invMM));
        const float ratio = maxScale / lepskiScale(k);
        const float sigmaFloor = perPhoton * (ratio * ratio);
        const float sigma = fmaxf(sigmaWelford, sigmaFloor) + 1e-7f;
        lowMax = fmaxf(lowMax, st.meanLum[k] - kappa * sigma);
        highMin = fminf(highMin, st.meanLum[k] + kappa * sigma);
        if (lowMax <= highMin)
            best = k;
        else
            break;
    }
    return best;
}

// The colours of LEPSKI_VIEW_HEATMAP, smallest scale to largest: blue, cyan,
// green, yellow, red.
__device__ __forceinline__ float3 lepskiScaleColor(int k)
{
    switch (k) {
    case 0:
        return make_float3(0.f, 0.f, 1.f);
    case 1:
        return make_float3(0.f, 1.f, 1.f);
    case 2:
        return make_float3(0.f, 1.f, 0.f);
    case 3:
        return make_float3(1.f, 1.f, 0.f);
    default:
        return make_float3(1.f, 0.f, 0.f);
    }
}

__global__ void lepskiGatherKernel(const __grid_constant__ LepskiParams L)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= (int)(L.width * L.height))
        return;

    const LepskiGatherPoint& gp = L.gather_points[px];
    const float3 frameDirect = gp.direct;
    LepskiLookup q;
    const bool gathers = lepskiGatherPixel<false>(L, gp, q);

    // This iteration's estimate at every scale: throughput * f_r times the
    // kernel-weighted flux (the flux is already per emitted path).  A pixel
    // that gathers nothing this iteration contributes its zeros — its sample
    // missed every diffuse surface, and the running mean is over the samples.
    const float3 weight = gp.weight;
    float3 pass[LEPSKI_K_SCALES];
#pragma unroll
    for (int k = 0; k < LEPSKI_K_SCALES; ++k)
        pass[k] = weight * q.flux[k];

    // The running estimate.  Iteration 1 starts it from nothing, so a reset
    // needs no clearing of the per-pixel state.
    const int M = L.iteration;
    PixelLepskiState st;
    float3 direct = frameDirect;
    if (M > 1) {
        st = L.pixel_state[px];
        direct += L.sum_direct[px];
    } else {
#pragma unroll
        for (int k = 0; k < LEPSKI_K_SCALES; ++k) {
            st.meanRGB[k] = make_float3(0.f, 0.f, 0.f);
            st.meanLum[k] = 0.f;
            st.m2Lum[k] = 0.f;
        }
        st.totalCountMax = 0u;
    }

    // Welford's update of the mean and of the sum of squared deviations.
    st.totalCountMax += q.accepted;
    const float invM = 1.f / (float)M;
#pragma unroll
    for (int k = 0; k < LEPSKI_K_SCALES; ++k) {
        st.meanRGB[k] += (pass[k] - st.meanRGB[k]) * invM;
        const float lum = lepskiLuminance(pass[k]);
        const float delta = lum - st.meanLum[k];
        st.meanLum[k] += delta * invM;
        st.m2Lum[k] += delta * (lum - st.meanLum[k]);
    }

    const int bestK = (L.view == LEPSKI_VIEW_FIXED) ? 2 : lepskiSelectScale(st, M, L.kappa);

    L.pixel_state[px] = st;
    L.sum_direct[px] = direct;

    // The image.  The exporter divides the accumulation buffer by the number
    // of frames, so it gets M times the running mean.
    const float fM = (float)M;
    float3 out;
    if (L.view == LEPSKI_VIEW_INDIRECT)
        out = fM * st.meanRGB[bestK];
    else if (L.view == LEPSKI_VIEW_HEATMAP)
        out = st.totalCountMax > 0u ? fM * lepskiScaleColor(bestK) : make_float3(0.f, 0.f, 0.f);
    else
        out = direct + fM * st.meanRGB[bestK];
    L.accum_buffer[px] = out;
    if (L.frame_radiance)
        L.frame_radiance[px] = frameDirect + pass[bestK];

    if (L.frame_buffer) {
        // clamped, gamma 2.0: the tone map of __raygen__rg and accumulateKernel
        const float mr = sqrtf(fminf(fmaxf(out.x * invM, 0.f), 1.f));
        const float mg = sqrtf(fminf(fmaxf(out.y * invM, 0.f), 1.f));
        const float mb = sqrtf(fminf(fmaxf(out.z * invM, 0.f), 1.f));
        L.frame_buffer[px] = make_uchar4((unsigned char)(mr * 255.f),
            (unsigned char)(mg * 255.f), (unsigned char)(mb * 255.f), 255u);
    }

    // --stats: one slot per pixel, so no atomics; -1 = no gather here.
    if (L.stat_tested) {
        L.stat_tested[px] = gathers ? q.tested : -1;
        L.stat_accepted[px] = gathers ? (int)q.accepted : -1;
        L.stat_cells[px] = gathers ? q.cells : -1;
    }
}

template <bool BRUTE>
__global__ void lepskiProbeKernel(const __grid_constant__ LepskiParams L, float3* __restrict__ out)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= (int)(L.width * L.height))
        return;
    const LepskiGatherPoint& gp = L.gather_points[px];
    LepskiLookup q;
    lepskiGatherPixel<BRUTE>(L, gp, q);
    float3 sum = make_float3(0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < LEPSKI_K_SCALES; ++k)
        sum += q.flux[k];
    out[px] = gp.weight * sum;
}

// --- Statistics --------------------------------------------------------------

__global__ void lepskiScaleHistogramKernel(const __grid_constant__ LepskiParams L,
    unsigned int* __restrict__ histogram)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= (int)(L.width * L.height))
        return;
    const PixelLepskiState& st = L.pixel_state[px];
    const int bin = st.totalCountMax > 0u ? lepskiSelectScale(st, L.iteration, L.kappa)
                                          : LEPSKI_K_SCALES;
    atomicAdd(&histogram[bin], 1u);
}

// One gather point's share of LepskiRadiusStats, and how two shares combine.
struct LepskiRadiusSample {
    float max_r;
    double sum_r;
    double sum_px;
    long long valid;
};

struct ToLepskiRadiusSample {
    float pixelAngle;
    __host__ __device__ LepskiRadiusSample operator()(const LepskiGatherPoint& gp) const
    {
        if (!(gp.r_base > 0.f))
            return LepskiRadiusSample { 0.f, 0.0, 0.0, 0 };
        const float r = lepskiScale(LEPSKI_K_SCALES - 1) * gp.r_base;
        const float w = gp.t * pixelAngle;
        return LepskiRadiusSample { r, (double)r, w > 0.f ? (double)(r / w) : 0.0, 1 };
    }
};

struct CombineLepskiRadiusSamples {
    __host__ __device__ LepskiRadiusSample operator()(const LepskiRadiusSample& a,
        const LepskiRadiusSample& b) const
    {
        return LepskiRadiusSample { fmaxf(a.max_r, b.max_r), a.sum_r + b.sum_r,
            a.sum_px + b.sum_px, a.valid + b.valid };
    }
};

typedef cub::TransformInputIterator<LepskiRadiusSample, ToLepskiRadiusSample,
    const LepskiGatherPoint*>
    LepskiRadiusIterator;

// --- Host --------------------------------------------------------------------

void lepskiFree(LepskiBuffers& b)
{
    auto release = [](void* p) {
        if (p)
            cudaFree(p);
    };
    release(b.gather_points);
    release(b.raw_photons);
    release(b.raw_hashes);
    release(b.sorted_photons);
    release(b.active_bits);
    release(b.counters);
    release(b.cell_count);
    release(b.cell_start);
    release(b.cell_cursor);
    release(b.pixel_state);
    release(b.sum_direct);
    release(b.cub_temp);
    release(b.stats_temp);
    release(b.stats_out);
    b = LepskiBuffers();
}

bool lepskiAllocate(LepskiBuffers& b, unsigned int width, unsigned int height,
    unsigned int maxStored)
{
    if (b.gather_points && b.width == width && b.height == height && b.max_stored >= maxStored)
        return false;
    lepskiFree(b);

    auto alloc = [&b](void* p, size_t bytes) {
        LEPSKI_CHECK(cudaMalloc((void**)p, bytes));
        b.bytes += bytes;
    };
    const size_t pixels = (size_t)width * height;
    const size_t tableBytes = kLepskiTableEntries * sizeof(unsigned int);
    alloc(&b.gather_points, pixels * sizeof(LepskiGatherPoint));
    alloc(&b.pixel_state, pixels * sizeof(PixelLepskiState));
    alloc(&b.sum_direct, pixels * sizeof(float3));
    alloc(&b.raw_photons, (size_t)maxStored * sizeof(PackedPhoton));
    alloc(&b.raw_hashes, (size_t)maxStored * sizeof(unsigned int));
    alloc(&b.sorted_photons, (size_t)maxStored * sizeof(PackedPhoton));
    alloc(&b.active_bits, kLepskiBitsetBytes);
    alloc(&b.counters, LEPSKI_COUNTERS * sizeof(unsigned int));
    alloc(&b.cell_count, tableBytes);
    alloc(&b.cell_start, tableBytes);
    alloc(&b.cell_cursor, tableBytes);

    // CUB tells how much working storage an algorithm needs when asked with
    // no storage at all.  Neither the table nor the image changes size while
    // these buffers live, so both are asked once, here.
    size_t tempBytes = 0;
    LEPSKI_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, b.cell_count, b.cell_start,
        (int)kLepskiTableEntries));
    alloc(&b.cub_temp, tempBytes);
    b.cub_temp_bytes = tempBytes;

    alloc(&b.stats_out, 64);
    size_t statsBytes = 0;
    LEPSKI_CHECK(cub::DeviceReduce::Reduce(nullptr, statsBytes,
        LepskiRadiusIterator(b.gather_points, ToLepskiRadiusSample { 0.f }),
        static_cast<LepskiRadiusSample*>(b.stats_out), (int)pixels,
        CombineLepskiRadiusSamples(), LepskiRadiusSample { 0.f, 0.0, 0.0, 0 }));
    alloc(&b.stats_temp, statsBytes);
    b.stats_temp_bytes = statsBytes;

    b.width = width;
    b.height = height;
    b.max_stored = maxStored;
    return true;
}

void lepskiBind(const LepskiBuffers& b, LepskiParams& L)
{
    L.gather_points = b.gather_points;
    L.raw_photons = b.raw_photons;
    L.raw_hashes = b.raw_hashes;
    L.max_stored = b.max_stored;
    L.active_bits = b.active_bits;
    L.counters = b.counters;
    L.cell_count = b.cell_count;
    L.cell_start = b.cell_start;
    L.cell_cursor = b.cell_cursor;
    L.sorted_photons = b.sorted_photons;
    L.pixel_state = b.pixel_state;
    L.sum_direct = b.sum_direct;
}

void lepskiClearCounters(const LepskiParams& L, cudaStream_t stream)
{
    LEPSKI_CHECK(cudaMemsetAsync(L.counters, 0, LEPSKI_COUNTERS * sizeof(unsigned int), stream));
}

void lepskiBeginIteration(const LepskiParams& L, cudaStream_t stream)
{
    LEPSKI_CHECK(cudaMemsetAsync(L.active_bits, 0, kLepskiBitsetBytes, stream));
    lepskiClearCounters(L, stream);
}

void lepskiBuildGrid(const LepskiParams& L, const LepskiBuffers& b, unsigned int stored,
    cudaStream_t stream)
{
    const size_t tableBytes = kLepskiTableEntries * sizeof(unsigned int);
    LEPSKI_CHECK(cudaMemsetAsync(L.cell_count, 0, tableBytes, stream));
    if (stored > 0)
        lepskiCountKernel<<<lepskiBlocksFor(stored), kLepskiBlock, 0, stream>>>(
            L.raw_hashes, (int)stored, L.cell_count);

    // Run on an empty map too: the gather reads cell_start either way.
    size_t tempBytes = b.cub_temp_bytes;
    LEPSKI_CHECK(cub::DeviceScan::ExclusiveSum(b.cub_temp, tempBytes, L.cell_count, L.cell_start,
        (int)kLepskiTableEntries, stream));

    if (stored > 0) {
        LEPSKI_CHECK(cudaMemcpyAsync(L.cell_cursor, L.cell_start, tableBytes,
            cudaMemcpyDeviceToDevice, stream));
        lepskiScatterKernel<<<lepskiBlocksFor(stored), kLepskiBlock, 0, stream>>>(
            L.raw_hashes, L.raw_photons, (int)stored, L.cell_cursor, L.sorted_photons);
    }
    LEPSKI_CHECK(cudaGetLastError());
}

void lepskiGather(const LepskiParams& L, cudaStream_t stream)
{
    const size_t pixels = (size_t)L.width * L.height;
    lepskiGatherKernel<<<lepskiBlocksFor(pixels), kLepskiBlock, 0, stream>>>(L);
    LEPSKI_CHECK(cudaGetLastError());
}

void lepskiProbe(const LepskiParams& L, bool brute, float3* out, cudaStream_t stream)
{
    const size_t pixels = (size_t)L.width * L.height;
    if (brute)
        lepskiProbeKernel<true><<<lepskiBlocksFor(pixels), kLepskiBlock, 0, stream>>>(L, out);
    else
        lepskiProbeKernel<false><<<lepskiBlocksFor(pixels), kLepskiBlock, 0, stream>>>(L, out);
    LEPSKI_CHECK(cudaGetLastError());
}

void lepskiRadiusStats(const LepskiParams& L, const LepskiBuffers& b, float pixelAngle,
    cudaStream_t stream, LepskiRadiusStats& out)
{
    LepskiRadiusSample* result = static_cast<LepskiRadiusSample*>(b.stats_out);
    size_t bytes = b.stats_temp_bytes;
    LEPSKI_CHECK(cub::DeviceReduce::Reduce(b.stats_temp, bytes,
        LepskiRadiusIterator(L.gather_points, ToLepskiRadiusSample { pixelAngle }), result,
        (int)((size_t)L.width * L.height), CombineLepskiRadiusSamples(),
        LepskiRadiusSample { 0.f, 0.0, 0.0, 0 }, stream));

    LepskiRadiusSample h;
    LEPSKI_CHECK(cudaMemcpyAsync(&h, result, sizeof(h), cudaMemcpyDeviceToHost, stream));
    LEPSKI_CHECK(cudaStreamSynchronize(stream));
    out.max_r = h.max_r;
    out.sum_r = h.sum_r;
    out.sum_px = h.sum_px;
    out.valid = h.valid;
}

void lepskiScaleHistogram(const LepskiParams& L, const LepskiBuffers& b,
    unsigned int histogram[LEPSKI_K_SCALES + 1], cudaStream_t stream)
{
    unsigned int* bins = static_cast<unsigned int*>(b.stats_out);
    const size_t bytes = (LEPSKI_K_SCALES + 1) * sizeof(unsigned int);
    const size_t pixels = (size_t)L.width * L.height;
    LEPSKI_CHECK(cudaMemsetAsync(bins, 0, bytes, stream));
    lepskiScaleHistogramKernel<<<lepskiBlocksFor(pixels), kLepskiBlock, 0, stream>>>(L, bins);
    LEPSKI_CHECK(cudaGetLastError());
    LEPSKI_CHECK(cudaMemcpyAsync(histogram, bins, bytes, cudaMemcpyDeviceToHost, stream));
    LEPSKI_CHECK(cudaStreamSynchronize(stream));
}

int lepskiKernelInfo(LepskiKernelInfo out[4])
{
    auto info = [](const char* name, const void* f) {
        cudaFuncAttributes a;
        LEPSKI_CHECK(cudaFuncGetAttributes(&a, f));
        return LepskiKernelInfo { name, a.numRegs, a.localSizeBytes };
    };
    out[0] = info("lepski_count", (const void*)lepskiCountKernel);
    out[1] = info("lepski_scatter", (const void*)lepskiScatterKernel);
    out[2] = info("lepski_gather", (const void*)lepskiGatherKernel);
    return 3;
}
