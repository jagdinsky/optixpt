// The CUDA side of RTPM (rtpm.h): its photon buffers, the marking of the
// culling mask, and the brute-force collection the structures are checked
// against.  The photon structures themselves are OptiX's (rtpm_host.h).
//
// A translation unit of its own, compiled by nvcc into the executable;
// nothing here is called by the other methods.
#include "rtpm.h"

#include <cstdio>
#include <cstdlib>

#define RTPM_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            std::fprintf(stderr, "[rtpm] %s:%d %s -> %s\n", __FILE__, \
                __LINE__, #call, cudaGetErrorString(_e));             \
            std::abort();                                             \
        }                                                             \
    } while (0)

static constexpr int kRtpmBlock = 256;
static inline int rtpmBlocksFor(size_t n) { return (int)((n + kRtpmBlock - 1) / kRtpmBlock); }

// Photon Culling, the camera side (paper sec. 3.7): every gather point marks
// the eight cells nearest to it.  A plain store of the constant 1 — every
// writer writes the same value, so no atomic is needed — and only by the
// threads that find the entry clear: thousands of pixels share a cell.
__global__ void rtpmMarkKernel(const __grid_constant__ Params P,
    unsigned char* __restrict__ mask, unsigned int bits, float cell, float3 origin)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (int)(P.width * P.height))
        return;
    const GatherPoint& gp = P.gather_points[j];
    if (!gp.valid)
        return;
    const int bx = rtpmBlockCoord(gp.pos.x, origin.x, cell);
    const int by = rtpmBlockCoord(gp.pos.y, origin.y, cell);
    const int bz = rtpmBlockCoord(gp.pos.z, origin.z, cell);
    for (int c = 0; c < 8; ++c) {
        const unsigned int i = rtpmCullIndex(bx + (c & 1), by + ((c >> 1) & 1), bz + (c >> 2), bits);
        if (!mask[i])
            mask[i] = 1;
    }
}

// The collection without the structures: every stored photon of both maps,
// for every pixel, through the intersection and any-hit programs' own tests
// and weights (rtpm_device.cu).
__global__ void rtpmBruteKernel(const __grid_constant__ Params P,
    const __grid_constant__ RtpmParams R, float3* __restrict__ out)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= (int)(P.width * P.height))
        return;

    const GatherPoint gp = P.gather_points[px];
    float3 radiance = gp.direct;
    if (gp.valid) {
        float3 irradiance = make_float3(0.f, 0.f, 0.f);
        for (int m = 0; m < RTPM_MAPS; ++m) {
            const RtpmMap& map = R.maps[m];
            const float r = gp.r * map.radius_scale;
            const float r2 = r * r;
            const float invR = 1.f / r;
            const float planeTol = P.photon_plane_tol * r;
            float3 flux = make_float3(0.f, 0.f, 0.f);
            for (unsigned int e = 0; e < map.stored; ++e) {
                const PackedPhoton& ph = map.photons[e];
                const float3 diff = subRn(ph.pos, gp.pos);
                const float d2 = dotRn(diff, diff);
                if (d2 > r2)
                    continue;
                if (!rtpmUsable(ph, diff, gp.normal, planeTol, P.photon_normal_tol,
                        gp.max_photon_depth))
                    continue;
                addWeighted(flux, rtpmWeight(d2, invR, R.uniform_kernel), ph.flux);
            }
            if (map.stored > 0u)
                irradiance += flux * rtpmNormalisation(r, R.uniform_kernel);
        }
        radiance += gp.weight * irradiance;
    }
    out[px] = radiance;
}

// --- Host --------------------------------------------------------------------

void rtpmFree(RtpmBuffers& b)
{
    auto release = [](void* p) {
        if (p)
            cudaFree(p);
    };
    for (int m = 0; m < RTPM_MAPS; ++m) {
        release(b.aabbs[m]);
        release(b.photons[m]);
    }
    release(b.counters);
    release(b.cull_mask);
    b = RtpmBuffers();
}

bool rtpmAllocate(RtpmBuffers& b, unsigned int causticCapacity, unsigned int globalCapacity,
    unsigned int cullBits)
{
    if (b.counters && b.capacity[RTPM_CAUSTIC] >= causticCapacity
        && b.capacity[RTPM_GLOBAL] >= globalCapacity && b.cull_bits == cullBits)
        return false;
    rtpmFree(b);

    auto alloc = [&b](void* p, size_t bytes) {
        RTPM_CHECK(cudaMalloc((void**)p, bytes));
        b.bytes += bytes;
    };
    const unsigned int capacity[RTPM_MAPS] = { causticCapacity, globalCapacity };
    for (int m = 0; m < RTPM_MAPS; ++m) {
        alloc(&b.aabbs[m], (size_t)capacity[m] * sizeof(OptixAabb));
        alloc(&b.photons[m], (size_t)capacity[m] * sizeof(PackedPhoton));
        b.capacity[m] = capacity[m];
    }
    alloc(&b.counters, RTPM_COUNTERS * sizeof(unsigned int));
    alloc(&b.cull_mask, (size_t)1 << cullBits);
    b.cull_bits = cullBits;
    return true;
}

void rtpmBind(const RtpmBuffers& b, RtpmParams& R)
{
    for (int m = 0; m < RTPM_MAPS; ++m) {
        R.maps[m].aabbs = b.aabbs[m];
        R.maps[m].photons = b.photons[m];
        R.maps[m].capacity = b.capacity[m];
    }
    R.counters = b.counters;
    R.cull_mask = b.cull_mask;
    R.cull_bits = b.cull_bits;
}

void rtpmClearCounters(const RtpmParams& R, cudaStream_t stream)
{
    RTPM_CHECK(cudaMemsetAsync(R.counters, 0, RTPM_COUNTERS * sizeof(unsigned int), stream));
}

void rtpmMarkCells(const Params& params, const RtpmParams& R, cudaStream_t stream)
{
    RTPM_CHECK(cudaMemsetAsync(R.cull_mask, 0, (size_t)1 << R.cull_bits, stream));
    const size_t pixels = (size_t)params.width * params.height;
    rtpmMarkKernel<<<rtpmBlocksFor(pixels), kRtpmBlock, 0, stream>>>(params, R.cull_mask,
        R.cull_bits, R.cull_cell, R.origin);
    RTPM_CHECK(cudaGetLastError());
}

void rtpmBruteCollect(const Params& params, const RtpmParams& R, float3* out, cudaStream_t stream)
{
    const size_t pixels = (size_t)params.width * params.height;
    rtpmBruteKernel<<<rtpmBlocksFor(pixels), kRtpmBlock, 0, stream>>>(params, R, out);
    RTPM_CHECK(cudaGetLastError());
}

int rtpmKernelInfo(RtpmKernelInfo out[2])
{
    cudaFuncAttributes a;
    RTPM_CHECK(cudaFuncGetAttributes(&a, (const void*)rtpmMarkKernel));
    out[0] = RtpmKernelInfo { "rtpm_mark", a.numRegs, a.localSizeBytes };
    return 1;
}
