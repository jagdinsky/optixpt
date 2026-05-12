#include <cuda_runtime.h>
#include <optix.h>

#include "params.h"

extern "C" __constant__ Params params;

// ─── float3 math helpers ──────────────────────────────────────────────────────

__device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(float t, float3 v)
{
    return make_float3(t * v.x, t * v.y, t * v.z);
}

__device__ __forceinline__ float3 operator*(float3 v, float t)
{
    return make_float3(t * v.x, t * v.y, t * v.z);
}

__device__ __forceinline__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator+=(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __forceinline__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float3 normalize(float3 v)
{
    float inv = 1.f / sqrtf(dot(v, v));
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

// ─── SBT record structs ───────────────────────────────────────────────────────

struct HitData {
    float3 albedo;
    float3 emission;
};
struct MissData {
    float3 bg_color;
};
struct RayGenData { };

// ─── PCG RNG ──────────────────────────────────────────────────────────────────

__device__ unsigned int pcg(unsigned int& state)
{
    state = state * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ float randf(unsigned int& rng)
{
    return (pcg(rng) & 0xFFFFFF) / float(0x1000000);
}

// ─── Cosine-weighted hemisphere sampling ─────────────────────────────────────

__device__ float3 cosineSampleHemisphere(float r1, float r2)
{
    float phi = 2.f * M_PIf * r1;
    float sinTheta = sqrtf(r2);
    float cosTheta = sqrtf(1.f - r2);
    return make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
}

// ─── ONB ─────────────────────────────────────────────────────────────────────

__device__ void onb(const float3& n, float3& t, float3& b)
{
    if (fabsf(n.x) > 0.9f)
        t = make_float3(0.f, 1.f, 0.f);
    else
        t = make_float3(1.f, 0.f, 0.f);
    b = normalize(cross(n, t));
    t = cross(b, n);
}

__device__ float3 toWorld(const float3& local, const float3& n, const float3& t, const float3& b)
{
    return local.x * t + local.y * b + local.z * n;
}

// ─── Raygen ───────────────────────────────────────────────────────────────────

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const int pixel = idx.y * params.width + idx.x;

    // Seed changes every frame AND every sample so we never repeat the same
    // random sequence when accumulating across frames.
    unsigned int rng = pixel * 1973u
        + (unsigned int)params.frame_index * 9277u
        + 4801u;

    float3 result = make_float3(0.f, 0.f, 0.f);

    for (int s = 0; s < params.samples_per_pixel; ++s) {
        float u = (idx.x + randf(rng)) / params.width;
        float v = (idx.y + randf(rng)) / params.height;

        float3 origin = params.cam_eye;
        float3 dir = normalize(params.cam_w
            + (2.f * u - 1.f) * params.cam_u
            + (2.f * v - 1.f) * params.cam_v);

        float3 radiance = make_float3(0.f, 0.f, 0.f);
        float3 attenuation = make_float3(1.f, 1.f, 1.f);
        bool done = false;

        for (int depth = 0; depth < params.max_depth && !done; ++depth) {
            unsigned int p0 = __float_as_uint(radiance.x);
            unsigned int p1 = __float_as_uint(radiance.y);
            unsigned int p2 = __float_as_uint(radiance.z);
            unsigned int p3 = __float_as_uint(attenuation.x);
            unsigned int p4 = __float_as_uint(attenuation.y);
            unsigned int p5 = __float_as_uint(attenuation.z);
            unsigned int p6 = 0u;
            unsigned int p7 = 0u, p8 = 0u, p9 = 0u; // hit_pos
            unsigned int p10 = 0u, p11 = 0u, p12 = 0u; // shading normal

            optixTrace(params.handle, origin, dir, 1e-3f, 1e16f, 0.f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                p0, p1, p2, p3, p4, p5, p6,
                p7, p8, p9, p10, p11, p12);

            radiance = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
            attenuation = make_float3(__uint_as_float(p3), __uint_as_float(p4), __uint_as_float(p5));
            done = (p6 == 1u);

            if (!done) {
                float3 hit_pos = make_float3(__uint_as_float(p7), __uint_as_float(p8), __uint_as_float(p9));
                float3 n = make_float3(__uint_as_float(p10), __uint_as_float(p11), __uint_as_float(p12));
                float3 t, b;
                onb(n, t, b);
                float3 local = cosineSampleHemisphere(randf(rng), randf(rng));
                origin = hit_pos;
                dir = normalize(toWorld(local, n, t, b));

                // Russian roulette after depth 3
                if (depth >= 3) {
                    float q = fmaxf(attenuation.x, fmaxf(attenuation.y, attenuation.z));
                    if (randf(rng) > q)
                        break;
                    attenuation.x /= q;
                    attenuation.y /= q;
                    attenuation.z /= q;
                }
            }
        }

        result.x += radiance.x;
        result.y += radiance.y;
        result.z += radiance.z;
    }

    // Average over samples_per_pixel for this frame
    result.x /= params.samples_per_pixel;
    result.y /= params.samples_per_pixel;
    result.z /= params.samples_per_pixel;

    // ── Progressive accumulation ──────────────────────────────────────────────
    // Add this frame's contribution to the running sum in accum_buffer.
    // frame_index is 0 on the first frame after a camera reset, so we
    // replace rather than add to avoid stale data from a previous view.
    float3 accumulated;
    if (params.frame_index == 0) {
        accumulated = result;
    } else {
        float3 prev = params.accum_buffer[pixel];
        accumulated = make_float3(prev.x + result.x,
            prev.y + result.y,
            prev.z + result.z);
    }
    params.accum_buffer[pixel] = accumulated;

    // Compute running mean: divide by total frames accumulated so far
    float n_frames = (float)(params.frame_index + 1);
    float3 mean = make_float3(accumulated.x / n_frames,
        accumulated.y / n_frames,
        accumulated.z / n_frames);

    // Gamma-correct (sRGB approx: gamma 2.2 ≈ sqrt) and clamp to [0,1]
    mean.x = sqrtf(fminf(fmaxf(mean.x, 0.f), 1.f));
    mean.y = sqrtf(fminf(fmaxf(mean.y, 0.f), 1.f));
    mean.z = sqrtf(fminf(fmaxf(mean.z, 0.f), 1.f));

    params.frame_buffer[pixel] = make_uchar4(
        (unsigned char)(mean.x * 255.f),
        (unsigned char)(mean.y * 255.f),
        (unsigned char)(mean.z * 255.f),
        255u);
}

// ─── Closesthit ───────────────────────────────────────────────────────────────

extern "C" __global__ void __closesthit__ch()
{
    const int triIdx = optixGetPrimitiveIndex();
    const float2 bary = optixGetTriangleBarycentrics();
    const float w = 1.f - bary.x - bary.y;
    const Triangle& tri = params.triangles[triIdx];

    float3 n = normalize(w * tri.n0 + bary.x * tri.n1 + bary.y * tri.n2);
    float3 ray_dir = optixGetWorldRayDirection();
    if (dot(n, ray_dir) > 0.f)
        n = make_float3(-n.x, -n.y, -n.z);

    const Material& mat = params.materials[tri.mat_id];

    float3 radiance = make_float3(__uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2()));
    float3 attenuation = make_float3(__uint_as_float(optixGetPayload_3()),
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()));

    radiance = radiance + attenuation * mat.emission;
    attenuation = attenuation * mat.albedo;

    float3 hit_pos = optixGetWorldRayOrigin()
        + optixGetRayTmax() * ray_dir;

    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
    optixSetPayload_3(__float_as_uint(attenuation.x));
    optixSetPayload_4(__float_as_uint(attenuation.y));
    optixSetPayload_5(__float_as_uint(attenuation.z));
    // p6 stays 0 — raygen will scatter and continue
    optixSetPayload_7(__float_as_uint(hit_pos.x));
    optixSetPayload_8(__float_as_uint(hit_pos.y));
    optixSetPayload_9(__float_as_uint(hit_pos.z));
    optixSetPayload_10(__float_as_uint(n.x));
    optixSetPayload_11(__float_as_uint(n.y));
    optixSetPayload_12(__float_as_uint(n.z));
}

// ─── Miss ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void __miss__ms()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    float3 radiance = make_float3(__uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2()));
    float3 attenuation = make_float3(__uint_as_float(optixGetPayload_3()),
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()));

    radiance.x += attenuation.x * data->bg_color.x;
    radiance.y += attenuation.y * data->bg_color.y;
    radiance.z += attenuation.z * data->bg_color.z;

    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
    optixSetPayload_6(1u); // signal "done"
}