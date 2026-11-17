#pragma once
#include "params.h"

// A photon in 32 bytes, and the packing of its two unit vectors.  Shared by
// the methods that keep photon maps of their own: the adaptive LoD hash-grid
// SPPM (sppm_lepski.h) and RTPM (rtpm.h).  The gathering families' Photon
// (params.h), 52 bytes, is not touched.

// A photon's bounce index shares a word with its normal, so it has six bits.
#define PACKED_PHOTON_MAX_DEPTH 63

// 32 bytes, so that four photons fill one 128-byte line.
//
// `flux` is the power the photon carries divided by the number of emitted
// paths — the renderer's convention (Photon::power) — so a sum of fluxes is
// already per emitted path.
//
// The normal has 2 x 13 bits instead of 2 x 16, and the six bits that frees
// hold the photon's bounce index.  A gather needs it: a photon stored at its
// bounce j joins a camera path of d + 1 vertices into a path of j + d + 1,
// and the path tracer stops at max_depth of them (see
// GatherPoint::max_photon_depth and cameraWalk in device.cu).  Without it a
// photon mapper integrates longer paths than the reference it is compared
// against.  13 bits resolve a normal to 0.02 degrees, and the normal is only
// compared against a threshold.
struct __align__(16) PackedPhoton
{
    float3 pos; // world-space hit position
    unsigned int wiOct; // unit vector back along the incoming path (2 x snorm16)
    float3 flux; // RGB power / emitted paths
    unsigned int nOct; // shading normal (2 x snorm13) | bounce index << 26
};
static_assert(sizeof(PackedPhoton) == 32, "PackedPhoton must be 32 bytes");

#ifdef __CUDACC__
#include "gather.h" // float3 operators, dot, normalize, length

// Octahedral unit vectors.  The fold maps the sphere onto the square
// [-1, 1]^2; what differs between the two packings below is only how many
// bits each coordinate gets.
__device__ __forceinline__ float2 packedOctFold(float3 v)
{
    const float invL1 = 1.0f / (fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + 1e-12f);
    float px = v.x * invL1;
    float py = v.y * invL1;
    if (v.z < 0.0f) {
        const float ox = (1.0f - fabsf(py)) * (px >= 0.0f ? 1.0f : -1.0f);
        const float oy = (1.0f - fabsf(px)) * (py >= 0.0f ? 1.0f : -1.0f);
        px = ox;
        py = oy;
    }
    return make_float2(fminf(fmaxf(px, -1.0f), 1.0f), fminf(fmaxf(py, -1.0f), 1.0f));
}

__device__ __forceinline__ float3 packedOctUnfold(float x, float y)
{
    const float z = 1.0f - fabsf(x) - fabsf(y);
    if (z < 0.0f) {
        const float ox = x;
        x += (ox >= 0.0f ? -1.0f : 1.0f) * (-z);
        y += (y >= 0.0f ? -1.0f : 1.0f) * (-z);
    }
    return normalize(make_float3(x, y, z));
}

// 2 x snorm16.  The int16 goes through uint16 before it is widened, so a
// negative coordinate does not sign-extend into the other half.
__device__ __forceinline__ unsigned int PackOctahedral(float3 v)
{
    const float2 p = packedOctFold(v);
    const short ix = (short)(p.x * 32767.0f);
    const short iy = (short)(p.y * 32767.0f);
    return ((unsigned int)(unsigned short)iy << 16) | (unsigned int)(unsigned short)ix;
}

__device__ __forceinline__ float3 UnpackOctahedral(unsigned int p)
{
    const short ix = (short)(unsigned short)(p & 0xFFFFu);
    const short iy = (short)(unsigned short)(p >> 16);
    return packedOctUnfold((float)ix * (1.0f / 32767.0f), (float)iy * (1.0f / 32767.0f));
}

// 2 x snorm13 and the photon's bounce index in the top six bits.
__device__ __forceinline__ unsigned int packNormalDepth(float3 n, int depth)
{
    const float2 p = packedOctFold(n);
    const int ix = (int)(p.x * 4095.0f);
    const int iy = (int)(p.y * 4095.0f);
    return ((unsigned int)depth << 26) | (((unsigned int)iy & 0x1FFFu) << 13)
        | ((unsigned int)ix & 0x1FFFu);
}

__device__ __forceinline__ float3 unpackNormal(unsigned int p)
{
    // shift the 13-bit field to the top, then back down arithmetically
    const int ix = (int)(p << 19) >> 19;
    const int iy = (int)(p << 6) >> 19;
    return packedOctUnfold((float)ix * (1.0f / 4095.0f), (float)iy * (1.0f / 4095.0f));
}

__device__ __forceinline__ int unpackDepth(unsigned int p) { return (int)(p >> 26); }
#endif // __CUDACC__
