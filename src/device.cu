#include <cuda_runtime.h>
#include <optix.h>

#include "params.h"

extern "C" __constant__ Params params;

// float3 helpers
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator*(float t, float3 v) { return make_float3(t * v.x, t * v.y, t * v.z); }
__device__ __forceinline__ float3 operator*(float3 v, float t) { return make_float3(t * v.x, t * v.y, t * v.z); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float3& operator+=(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __forceinline__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ __forceinline__ float3 normalize(float3 v)
{
    float inv = 1.f / sqrtf(dot(v, v));
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}
__device__ __forceinline__ float length(float3 v) { return sqrtf(dot(v, v)); }

// SBT structs
struct MissData {
    float3 bg_color;
};
struct RayGenData { };

// PCG RNG
__device__ unsigned int pcg(unsigned int& s)
{
    s = s * 747796405u + 2891336453u;
    unsigned int w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
__device__ float randf(unsigned int& rng) { return (pcg(rng) & 0xFFFFFF) / float(0x1000000); }

// Cosine-weighted hemisphere sample
__device__ float3 cosineSampleHemisphere(float r1, float r2)
{
    float phi = 2.f * M_PIf * r1, s = sqrtf(r2);
    return make_float3(cosf(phi) * s, sinf(phi) * s, sqrtf(1.f - r2));
}
__device__ __forceinline__ float pdfCosineHemisphere(float cosT) { return fmaxf(cosT, 0.f) * M_1_PIf; }

// ONB
__device__ void onb(const float3& n, float3& t, float3& b)
{
    t = (fabsf(n.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    b = normalize(cross(n, t));
    t = cross(b, n);
}
__device__ float3 toWorld(float3 l, float3 n, float3 t, float3 b) { return l.x * t + l.y * b + l.z * n; }

// Schlick Fresnel: R0 = ((1-ior)/(1+ior))^2
__device__ float schlick(float cosTheta, float ior)
{
    float r0 = (1.f - ior) / (1.f + ior);
    r0 *= r0;
    return r0 + (1.f - r0) * powf(1.f - cosTheta, 5.f);
}

// Safe refract — returns false on total internal reflection
__device__ bool refractDir(float3 d, float3 n, float eta, float3& refracted)
{
    float cosi = dot(-d, n);
    float sin2t = eta * eta * (1.f - cosi * cosi);
    if (sin2t > 1.f)
        return false;
    refracted = normalize(eta * d + (eta * cosi - sqrtf(1.f - sin2t)) * n);
    return true;
}

// Texture sampling
__device__ __forceinline__ float3 sampleTex(cudaTextureObject_t tex, float u, float v, float3 fallback)
{
    if (tex == 0)
        return fallback;
    float4 s = tex2D<float4>(tex, u, v);
    return make_float3(s.x, s.y, s.z);
}

// Triangle point sampling
__device__ float3 sampleTriangle(float3 v0, float3 v1, float3 v2, float r1, float r2)
{
    float su = sqrtf(r1), u = 1.f - su, v = r2 * su;
    return u * v0 + v * v1 + (1.f - u - v) * v2;
}

// Shadow ray (ray type 1)
__device__ bool isVisible(float3 origin, float3 target)
{
    float3 d = target - origin;
    float tmax = length(d) - 2e-3f;
    if (tmax <= 0.f)
        return false;
    d = normalize(d);
    unsigned int vis = 0u;
    optixTrace(params.handle, origin, d, 1e-3f, tmax, 0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0, 1, 1, vis);
    return (vis == 1u);
}

// MIS power heuristic (β=2)
__device__ __forceinline__ float misWeight(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA, b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-10f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Payload layout  (numPayloadValues = 16)
//  ─────────────────────────────────────────────────────────────────────────────
//  p0-p2   radiance.xyz            rw  raygen ↔ CH/miss
//  p3-p5   throughput.xyz (in)     rw  raygen -> CH;  CH reads for emission
//          albedo.xyz      (out)   rw  CH -> raygen;  raygen does BxDF eval
//  p6      nextBsdfPdf     (in)    r   raygen -> CH (for MIS of hit emission)
//          (set 0u by CH on return, not used by raygen as done-flag anymore)
//  p7-p9   hitpos.xyz              out CH -> raygen
//  p10-p12 shading normal.xyz      out CH -> raygen  (face-forward)
//  p13     matType (lo 8 bits)     out CH -> raygen
//          outsideFlag (bit 8)     out CH -> raygen  (for glass eta selection)
//  p14     mat_id (uint)           out CH -> raygen  (for ior lookup)
//  p15     done flag (0=hit,1=miss) out CH/miss -> raygen
// ─────────────────────────────────────────────────────────────────────────────

// Raygen
extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const int pixel = idx.y * params.width + idx.x;

    unsigned int rng = pixel * 1973u + (unsigned int)(params.frame_index) * 9277u + 4801u;

    float3 result = make_float3(0.f, 0.f, 0.f);

    for (int s = 0; s < params.samples_per_pixel; s++) {
        float pu = (idx.x + randf(rng)) / params.width;
        float pv = (idx.y + randf(rng)) / params.height;

        float3 origin = params.cam_eye;
        float3 dir = normalize(params.cam_w
            + (2.f * pu - 1.f) * params.cam_u
            + (2.f * pv - 1.f) * params.cam_v);
        float3 radiance = make_float3(0.f, 0.f, 0.f);
        float3 throughput = make_float3(1.f, 1.f, 1.f);
        float nextBsdfPdf = 0.f; // 0 = camera ray -> full emission weight

        for (int depth = 0; depth < params.max_depth; depth++) {

            unsigned int p0 = __float_as_uint(radiance.x);
            unsigned int p1 = __float_as_uint(radiance.y);
            unsigned int p2 = __float_as_uint(radiance.z);
            unsigned int p3 = __float_as_uint(throughput.x);
            unsigned int p4 = __float_as_uint(throughput.y);
            unsigned int p5 = __float_as_uint(throughput.z);
            unsigned int p6 = __float_as_uint(nextBsdfPdf);
            unsigned int p7 = 0u, p8 = 0u, p9 = 0u;
            unsigned int p10 = 0u, p11 = 0u, p12 = 0u;
            unsigned int p13 = 0u, p14 = 0u, p15 = 0u;

            optixTrace(
                params.handle, origin, dir,
                1e-3f, 1e16f, 0.f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                p0, p1, p2, p3, p4, p5, p6,
                p7, p8, p9, p10, p11, p12,
                p13, p14, p15);

            radiance = make_float3(__uint_as_float(p0),
                __uint_as_float(p1),
                __uint_as_float(p2));

            if (p15 == 1u)
                break; // miss set done

            unsigned int matType = p13 & 0xFFu;
            bool outsideFlag = (p13 >> 8u) & 1u;
            unsigned int mat_id = p14;

            float3 hitpos = make_float3(__uint_as_float(p7),
                __uint_as_float(p8),
                __uint_as_float(p9));
            float3 n = make_float3(__uint_as_float(p10),
                __uint_as_float(p11),
                __uint_as_float(p12));
            // albedo written by CH into p3-p5 for all matTypes
            float3 albedo = make_float3(__uint_as_float(p3),
                __uint_as_float(p4),
                __uint_as_float(p5));

            // BxDF dispatch
            if (matType == MAT_DIFFUSE) {

                // Russian Roulette
                if (depth >= 3) {
                    float q = fmaxf(albedo.x, fmaxf(albedo.y, albedo.z));
                    q = fmaxf(q, 0.05f);
                    if (randf(rng) > q)
                        break;
                    albedo = albedo * (1.f / q);
                }

                // NEE — uniform light sampling + MIS
                if (params.num_lights > 0) {
                    int lIdx = min((int)(randf(rng) * params.num_lights),
                        params.num_lights - 1);
                    const EmissiveTriangle& lt = params.lights[lIdx];
                    float3 lp = sampleTriangle(lt.v0, lt.v1, lt.v2, randf(rng), randf(rng));
                    float3 toLight = lp - hitpos;
                    float dist2 = dot(toLight, toLight);
                    float dist = sqrtf(dist2);
                    float3 wi = toLight * (1.f / dist);
                    float cosN = dot(n, wi);
                    float3 ln = normalize(cross(lt.v1 - lt.v0, lt.v2 - lt.v0));
                    float cosLight = fabsf(dot(ln, wi));
                    if (cosN > 0.f && cosLight > 1e-4f && isVisible(hitpos, lp)) {
                        float pLightArea = 1.f / (params.num_lights * lt.area);
                        float pLightSA = pLightArea * dist2 / cosLight;
                        float pBsdfSA = pdfCosineHemisphere(cosN);
                        float wNEE = misWeight(pLightSA, pBsdfSA);
                        radiance += throughput * albedo * M_1_PIf
                            * lt.emission * cosN * (1.0f / pLightSA) * wNEE;
                    }
                }

                // Cosine-weighted BSDF sample
                float3 tvec, bvec;
                onb(n, tvec, bvec);
                float3 local = cosineSampleHemisphere(randf(rng), randf(rng));
                float3 newdir = normalize(toWorld(local, n, tvec, bvec));
                float cosTheta = fmaxf(dot(n, newdir), 0.f);

                throughput = throughput * albedo; // pdf/cos cancel for cosine-weighted
                nextBsdfPdf = pdfCosineHemisphere(cosTheta);
                origin = hitpos;
                dir = newdir;

            } else if (matType == MAT_MIRROR) {
                // Perfect specular reflection
                // Fix: throughput gets ALL 3 albedo channels (was missing .z)
                float3 newdir = normalize(dir - 2.f * dot(dir, n) * n);
                throughput = throughput * albedo; // tint: full RGB
                nextBsdfPdf = 0.f; // delta -> MIS uses full emission weight
                origin = hitpos + 1e-3f * newdir;
                dir = newdir;

            } else {
                // Glass: stochastic Fresnel
                // Fix: use randf(rng) for Fresnel decision (was deterministic fr>0.5)
                float ior = params.materials[mat_id].ior;
                // outsideFlag: CH set 1 if dot(raydir, geo_n) < 0 (entering medium)
                float eta = outsideFlag ? (1.f / ior) : ior;
                // n is already face-forward (opposite to incoming dir) from CH
                float cosT = fabsf(dot(-dir, n));
                float fr = schlick(cosT, ior);

                float3 newdir;
                bool doReflect = !refractDir(dir, n, eta, newdir); // TIR?
                if (!doReflect)
                    doReflect = (randf(rng) < fr); // stochastic Fresnel
                if (doReflect)
                    newdir = normalize(dir - 2.f * dot(dir, n) * n);

                // throughput unchanged — the stochastic choice is unbiased
                nextBsdfPdf = 0.f;
                origin = hitpos + 1e-3f * newdir;
                dir = newdir;
            }
        }

        result += radiance;
    }

    result.x /= params.samples_per_pixel;
    result.y /= params.samples_per_pixel;
    result.z /= params.samples_per_pixel;

    // Temporal accumulation
    float3 accumulated = result;
    if (params.frame_index > 0) {
        float3 prev = params.accum_buffer[pixel];
        accumulated = make_float3(prev.x + result.x, prev.y + result.y, prev.z + result.z);
    }
    params.accum_buffer[pixel] = accumulated;

    float nf = float(params.frame_index + 1);
    float3 mean = make_float3(accumulated.x / nf, accumulated.y / nf, accumulated.z / nf);

    // Gamma 2.0 (sqrt)
    mean.x = sqrtf(fminf(fmaxf(mean.x, 0.f), 1.f));
    mean.y = sqrtf(fminf(fmaxf(mean.y, 0.f), 1.f));
    mean.z = sqrtf(fminf(fmaxf(mean.z, 0.f), 1.f));

    params.frame_buffer[pixel] = make_uchar4(
        (unsigned char)(mean.x * 255.f),
        (unsigned char)(mean.y * 255.f),
        (unsigned char)(mean.z * 255.f),
        255u);
}

extern "C" __global__ void __raygen__photon()
{
    const int photon_id = optixGetLaunchIndex().x;
    unsigned int rng = photon_id * 2791u
        + (unsigned int)params.frame_index * 5923u + 1847u;

    if (params.num_lights == 0)
        return;
    int lIdx = min((int)(randf(rng) * params.num_lights), params.num_lights - 1);
    const EmissiveTriangle& lt = params.lights[lIdx];

    float3 origin = sampleTriangle(lt.v0, lt.v1, lt.v2, randf(rng), randf(rng));
    float3 ln = normalize(cross(lt.v1 - lt.v0, lt.v2 - lt.v0));
    float3 t, b;
    onb(ln, t, b);
    float3 local = cosineSampleHemisphere(randf(rng), randf(rng));
    float3 dir = normalize(toWorld(local, ln, t, b));

    float3 power = lt.emission * lt.area * M_PIf
        * (float)params.num_lights
        * (1.0f / (float)params.num_photons);

    for (int depth = 0; depth < params.max_depth; depth++) {
        // Payload: same as CH layout
        unsigned int p0 = 0u, p1 = 0u, p2 = 0u; // radiance (unused here)
        unsigned int p3 = __float_as_uint(1.f); // throughput.x in (CH игнорирует)
        unsigned int p4 = __float_as_uint(1.f); // throughput.y in
        unsigned int p5 = __float_as_uint(1.f); // throughput.z in
        unsigned int p6 = 0u; // nextBsdfPdf in
        unsigned int p7 = 0u, p8 = 0u, p9 = 0u; // hitpos (out)
        unsigned int p10 = 0u, p11 = 0u, p12 = 0u; // normal (out)
        unsigned int p13 = 0u; // matType | outsideFlag (out)
        unsigned int p14 = 0u; // mat_id (out)
        unsigned int p15 = 0u; // done (out)

        optixTrace(
            params.handle, origin, dir,
            1e-3f, 1e16f, 0.f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1, p2, p3, p4, p5, p6,
            p7, p8, p9, p10, p11, p12,
            p13, p14, p15);

        // Read from payload
        if (p15 == 1u)
            break; // miss: p15 = done

        unsigned int matType = p13 & 0xFFu;
        unsigned int mat_id = p14;
        bool outsideFlag = (p13 >> 8u) & 1u;

        float3 hitpos = make_float3(__uint_as_float(p7), // p7-p9!
            __uint_as_float(p8),
            __uint_as_float(p9));
        float3 n = make_float3(__uint_as_float(p10), // p10-p12!
            __uint_as_float(p11),
            __uint_as_float(p12));
        float3 albedo = make_float3(__uint_as_float(p3), // p3-p5!
            __uint_as_float(p4),
            __uint_as_float(p5));

        if (matType == MAT_DIFFUSE) {
            float survive = fmaxf(albedo.x, fmaxf(albedo.y, albedo.z));
            survive = fmaxf(survive, 0.05f);

            int slot = atomicAdd(params.photon_count, 1);
            if (slot < params.num_photons) {
                Photon ph;
                ph.pos = hitpos;
                ph.power = power;
                ph.dir = -dir;
                ph._pad = 0.f;
                params.photon_map[slot] = ph;
            }

            if (randf(rng) > survive)
                break;
            power = power * albedo * (1.f / survive);

            float3 tv, bv;
            onb(n, tv, bv);
            float3 loc = cosineSampleHemisphere(randf(rng), randf(rng));
            dir = normalize(toWorld(loc, n, tv, bv));
            origin = hitpos;

        } else if (matType == MAT_MIRROR) {
            dir = normalize(dir - 2.f * dot(dir, n) * n);
            power = power * albedo;
            origin = hitpos + 1e-3f * dir;

        } else { // MAT_GLASS
            float ior = params.materials[mat_id].ior;
            float eta = outsideFlag ? (1.f / ior) : ior;
            float cosT = fabsf(dot(-dir, n));
            float fr = schlick(cosT, ior);
            float3 newdir;
            bool doRefl = !refractDir(dir, n, eta, newdir);
            if (!doRefl)
                doRefl = (randf(rng) < fr);
            if (doRefl)
                newdir = normalize(dir - 2.f * dot(dir, n) * n);
            dir = newdir;
            origin = hitpos + 1e-3f * dir;
        }
    }
}

extern "C" __global__ void __raygen__gather()
{
    const uint3 idx = optixGetLaunchIndex();
    const int pixel = idx.y * params.width + idx.x;
    unsigned int rng = pixel * 1973u + (unsigned int)params.frame_index * 9277u + 4801u;

    float pu = (idx.x + randf(rng)) / params.width;
    float pv = (idx.y + randf(rng)) / params.height;

    float3 origin = params.cam_eye;
    float3 dir = normalize(params.cam_w
        + (2.f * pu - 1.f) * params.cam_u
        + (2.f * pv - 1.f) * params.cam_v);

    float3 result = make_float3(0.f, 0.f, 0.f);
    float3 path_throughput = make_float3(1.f, 1.f, 1.f);

    for (int depth = 0; depth < params.max_depth; depth++) {

        // p0-p2   radiance (CH writes emission)
        // p3-p5   albedo   (CH writes albedo of all matType)
        // p6      bsdfPdf  (CH writes 0u, not needed for gathering)
        // p7-p9   hitpos   (CH writes)
        // p10-p12 normal   (CH writes, face-forward)
        // p13     matType[7:0] | outsideFlag[8]
        // p14     mat_id
        // p15     done (0=hit, 1=miss)
        unsigned int p0 = 0u, p1 = 0u, p2 = 0u;
        unsigned int p3 = __float_as_uint(1.f); // throughput.x -> CH reads for emission MIS
        unsigned int p4 = __float_as_uint(1.f); // throughput.y
        unsigned int p5 = __float_as_uint(1.f); // throughput.z
        unsigned int p6 = 0u; // nextBsdfPdf = 0 -> CH gives full emission weight
        unsigned int p7 = 0u, p8 = 0u, p9 = 0u;
        unsigned int p10 = 0u, p11 = 0u, p12 = 0u;
        unsigned int p13 = 0u, p14 = 0u, p15 = 0u;

        optixTrace(params.handle, origin, dir,
            1e-3f, 1e16f, 0.f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1, p2, p3, p4, p5,
            p6, p7, p8, p9, p10, p11,
            p12, p13, p14, p15);

        if (p15 == 1u) {
            // Miss —> sky (CH/miss calculates sky radiance in p0-p2 with throughput=1)
            result += path_throughput * make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
            break;
        }

        // Read payload
        unsigned int matType = p13 & 0xFFu;
        bool outsideFlag = (p13 >> 8u) & 1u;
        unsigned int mat_id = p14;

        float3 hitpos = make_float3(__uint_as_float(p7),
            __uint_as_float(p8),
            __uint_as_float(p9));
        float3 n = make_float3(__uint_as_float(p10),
            __uint_as_float(p11),
            __uint_as_float(p12));
        float3 albedo = make_float3(__uint_as_float(p3),
            __uint_as_float(p4),
            __uint_as_float(p5));

        // Emission from light source (CH wrote to p0-p2 with throughput=1)
        float3 emission = make_float3(__uint_as_float(p0),
            __uint_as_float(p1),
            __uint_as_float(p2));
        result += path_throughput * emission;

        if (matType == MAT_DIFFUSE) {
            // Photon gathering (brute-force)
            float r2 = params.gather_radius * params.gather_radius;
            int stored = min(*params.photon_count, params.num_photons);

            float3 irradiance = make_float3(0.f, 0.f, 0.f);
            int count = 0;

            for (int i = 0; i < stored; i++) {
                const Photon& ph = params.photon_map[i];
                float3 diff = ph.pos - hitpos;
                float d2 = dot(diff, diff);
                if (d2 > r2)
                    continue;

                // Cone filter k=1: w = 1 - dist/(k*r)
                float dist = sqrtf(d2);
                float weight = 1.f - dist / params.gather_radius;
                irradiance += weight * ph.power;
                count++;
            }

            // Normalization of the conical filter k=1: π*r²/3
            if (count > 0) {
                float norm = 3.f / (M_PIf * r2);
                irradiance = irradiance * norm;
            }

            // Lambertian BRDF: fr = albedo/π, times π from gathering -> albedo
            result += path_throughput * albedo * M_1_PIf * irradiance;
            break; // PM: only first diffuse bounce

        } else if (matType == MAT_MIRROR) {
            path_throughput = path_throughput * albedo;
            dir = normalize(dir - 2.f * dot(dir, n) * n);
            origin = hitpos + 1e-3f * dir;

        } else { // MAT_GLASS
            float ior = params.materials[mat_id].ior;
            float eta = outsideFlag ? (1.f / ior) : ior;
            float cosT = fabsf(dot(-dir, n));
            float fr = schlick(cosT, ior);
            float3 newdir;
            bool doRefl = !refractDir(dir, n, eta, newdir);
            if (!doRefl)
                doRefl = (randf(rng) < fr);
            if (doRefl)
                newdir = normalize(dir - 2.f * dot(dir, n) * n);
            dir = newdir;
            origin = hitpos + 1e-3f * dir;
        }
    }

    // Temporal accumulation
    float3 prev = (params.frame_index == 0)
        ? make_float3(0.f, 0.f, 0.f)
        : params.accum_buffer[pixel];
    float3 acc = make_float3(prev.x + result.x,
        prev.y + result.y,
        prev.z + result.z);
    params.accum_buffer[pixel] = acc;

    float nf = float(params.frame_index + 1);
    float3 mean = make_float3(acc.x / nf, acc.y / nf, acc.z / nf);
    mean.x = sqrtf(fminf(fmaxf(mean.x, 0.f), 1.f));
    mean.y = sqrtf(fminf(fmaxf(mean.y, 0.f), 1.f));
    mean.z = sqrtf(fminf(fmaxf(mean.z, 0.f), 1.f));
    params.frame_buffer[pixel] = make_uchar4(
        (unsigned char)(mean.x * 255.f),
        (unsigned char)(mean.y * 255.f),
        (unsigned char)(mean.z * 255.f), 255u);
}

// Closesthit — geometry + emission only, NO BxDF logic
extern "C" __global__ void __closesthit__ch()
{
    const int triIdx = optixGetPrimitiveIndex();
    const float2 bary = optixGetTriangleBarycentrics();
    const float w = 1.f - bary.x - bary.y;
    const Triangle& tri = params.triangles[triIdx];

    float3 raydir = optixGetWorldRayDirection();
    float3 n = normalize(w * tri.n0 + bary.x * tri.n1 + bary.y * tri.n2);
    bool outside = (dot(raydir, n) < 0.f); // true = ray enters from outside
    if (!outside)
        n = make_float3(-n.x, -n.y, -n.z); // flip to face-forward

    float uvu = w * tri.uv0.x + bary.x * tri.uv1.x + bary.y * tri.uv2.x;
    float uvv = w * tri.uv0.y + bary.x * tri.uv1.y + bary.y * tri.uv2.y;

    const Material& mat = params.materials[tri.mat_id];

    // Resolve albedo (texture * factor or just factor)
    float3 albedo;
    if (mat.base_color_tex != 0) {
        float4 s = tex2D<float4>(mat.base_color_tex, uvu, uvv);
        albedo = make_float3(s.x * mat.albedo.x, s.y * mat.albedo.y, s.z * mat.albedo.z);
    } else {
        albedo = mat.albedo;
    }

    // Resolve emission
    float3 emission;
    if (mat.emissive_tex != 0) {
        float4 s = tex2D<float4>(mat.emissive_tex, uvu, uvv);
        emission = make_float3(s.x * mat.emission.x, s.y * mat.emission.y, s.z * mat.emission.z);
    } else {
        emission = mat.emission;
    }

    float3 hitpos = optixGetWorldRayOrigin() + optixGetRayTmax() * raydir;

    // Read radiance + throughput + bsdfpdf from incoming payload
    float3 radiance = make_float3(__uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2()));
    float3 throughput = make_float3(__uint_as_float(optixGetPayload_3()),
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()));
    float bsdfpdf = __uint_as_float(optixGetPayload_6());

    // Emission accounting (MIS)
    if (emission.x + emission.y + emission.z > 0.f) {
        if (bsdfpdf < 1e-10f) {
            // Camera ray or delta (mirror/glass): full emission weight
            radiance += throughput * emission;
        } else {
            // BSDF-sampled bounce: MIS vs NEE pdf
            float pLightSA = 0.f;
            for (int li = 0; li < params.num_lights; li++) {
                if (params.lights[li].tri_idx == triIdx) {
                    float3 toSurf = hitpos - optixGetWorldRayOrigin();
                    float dist2 = dot(toSurf, toSurf);
                    float3 ln = normalize(cross(params.lights[li].v1 - params.lights[li].v0,
                        params.lights[li].v2 - params.lights[li].v0));
                    float cosL = fabsf(dot(ln, -raydir));
                    float parea = 1.f / (params.num_lights * params.lights[li].area);
                    pLightSA = parea * dist2 / fmaxf(cosL, 1e-4f);
                    break;
                }
            }
            radiance += throughput * emission * misWeight(bsdfpdf, pLightSA);
        }
    }

    // Write outputs
    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));

    // p3-p5: albedo for ALL matTypes (raygen performs BxDF eval)
    optixSetPayload_3(__float_as_uint(albedo.x));
    optixSetPayload_4(__float_as_uint(albedo.y));
    optixSetPayload_5(__float_as_uint(albedo.z));

    optixSetPayload_6(0u); // clear bsdfpdf (raygen sets nextBsdfPdf after BxDF)

    optixSetPayload_7(__float_as_uint(hitpos.x));
    optixSetPayload_8(__float_as_uint(hitpos.y));
    optixSetPayload_9(__float_as_uint(hitpos.z));

    optixSetPayload_10(__float_as_uint(n.x));
    optixSetPayload_11(__float_as_uint(n.y));
    optixSetPayload_12(__float_as_uint(n.z));

    // pack matType (lo 8) and outside flag (bit 8) into p13
    unsigned int outsideBit = outside ? (1u << 8u) : 0u;
    optixSetPayload_13((unsigned int)(mat.matType & 0xFF) | outsideBit);

    // mat_id for ior lookup in raygen
    optixSetPayload_14((unsigned int)tri.mat_id);

    // done = 0 (valid hit)
    optixSetPayload_15(0u);
}

// Primary miss: sky gradient
extern "C" __global__ void __miss__ms()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    float3 raydir = optixGetWorldRayDirection();

    float t = fminf(fmaxf(0.5f * (raydir.y + 1.f), 0.f), 1.f);
    float3 sky = make_float3((1.f - t) * 1.0f + t * 0.4f,
                     (1.f - t) * 0.95f + t * 0.6f,
                     (1.f - t) * 0.85f + t * 1.0f)
        * data->bg_color.x;

    float3 radiance = make_float3(__uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2()));
    float3 throughput = make_float3(__uint_as_float(optixGetPayload_3()),
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()));

    radiance += throughput * sky;

    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
    optixSetPayload_15(1u); // done = true
}

// Shadow miss
extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(1u); // unoccluded = true
}