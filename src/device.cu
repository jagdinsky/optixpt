#include <cuda_runtime.h>
#include <optix.h>

#include "params.h"

extern "C" __constant__ Params params;

// helpers
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
    float len2 = dot(v, v);
    if (len2 < 1e-20f) return make_float3(0.f, 0.f, 0.f);
    float inv = rsqrtf(len2);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}
__device__ __forceinline__ float length(float3 v) { return sqrtf(dot(v, v)); }

__device__ __forceinline__ float srgbToLinear(float c)
{
    return (c <= 0.04045f) ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float3 sampleTexSRGB(cudaTextureObject_t tex,
    float u, float v,
    float3 fallback)
{
    if (tex == 0)
        return fallback;
    float4 s = tex2D<float4>(tex, u, v);
    return make_float3(srgbToLinear(s.x),
        srgbToLinear(s.y),
        srgbToLinear(s.z));
}

// Decides whether a stored photon may contribute to the estimate at `x`.
// The plain 3D distance test is not enough: a sphere of radius r around a point
// near a corner also encloses photons that landed on the *neighbouring* wall,
// and those are what leak across edges and show up as blotches.  Requiring the
// photon to sit near the tangent plane turns the sphere into a thin disc, and
// requiring a matching normal keeps the two sides of a thin wall apart.
__device__ __forceinline__ bool photonUsable(const Photon& ph, float3 x, float3 n,
    float r2, float planeTol, float normalTol, int maxPhotonDepth, float& d2)
{
    // Path length first, because it is the cheapest test and because getting it
    // wrong is not a matter of a few stray photons: without it the camera walk
    // and the photon walk each get their own max_depth, so photon mapping
    // integrates paths up to twice as long as the path tracer does and the two
    // modes stop solving the same problem.  See __raygen__gather for the budget.
    if (ph.depth > maxPhotonDepth)
        return false;
    float3 diff = ph.pos - x;
    d2 = dot(diff, diff);
    if (d2 > r2)
        return false;
    if (dot(n, ph.normal) < normalTol)
        return false;
    if (fabsf(dot(diff, n)) > planeTol)
        return false;
    // The photon must have arrived on the side we are shading.
    if (dot(n, ph.dir) <= 1e-4f)
        return false;
    return true;
}

// Cone-filtered flux within `r`, and how many photons contributed.  The count
// is what lets the caller adapt the radius to the local photon density.
__device__ void gatherPhotonsGrid(float3 hitpos, float3 n, float r, float r2,
    int maxPhotonDepth, float3& flux, int& count)
{
    const PhotonGrid& g = params.grid;
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = params.photon_plane_tol * r;
    const float normalTol = params.photon_normal_tol;
    const float invR = 1.f / r;

    // hitpos in grid coordinates
    int cx = (int)floorf((hitpos.x - g.aabb_min.x) / g.cell_size);
    int cy = (int)floorf((hitpos.y - g.aabb_min.y) / g.cell_size);
    int cz = (int)floorf((hitpos.z - g.aabb_min.z) / g.cell_size);

    // go over neighboring cells (3x3x3); cell_size is the maximum gather
    // radius, so this neighbourhood always covers the search sphere
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = cz + dz;
        if (nz < 0 || nz >= g.dims.z)
            continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy;
            if (ny < 0 || ny >= g.dims.y)
                continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx;
                if (nx < 0 || nx >= g.dims.x)
                    continue;

                int cell = nz * g.dims.y * g.dims.x + ny * g.dims.x + nx;
                int start = g.cell_start[cell];
                int cellCount = g.cell_count[cell];

                for (int k = 0; k < cellCount; ++k) {
                    int pid = g.grid_photon_ids[start + k];
                    const Photon& ph = params.photon_map[pid];

                    float d2;
                    if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                            maxPhotonDepth, d2))
                        continue;

                    float weight = 1.f - sqrtf(d2) * invR; // cone filter k=1
                    flux += weight * ph.power;
                    ++count;
                }
            }
        }
    }
}

__device__ void gatherBruteForce(float3 hitpos, float3 n, float r, float r2,
    int maxPhotonDepth, float3& flux, int& count)
{
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = params.photon_plane_tol * r;
    const float normalTol = params.photon_normal_tol;
    const float invR = 1.f / r;

    int stored = min(*params.photon_count, params.photon_capacity);
    for (int i = 0; i < stored; i++) {
        const Photon& ph = params.photon_map[i];

        float d2;
        if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                maxPhotonDepth, d2))
            continue;

        float weight = 1.f - sqrtf(d2) * invR;
        flux += weight * ph.power;
        ++count;
    }
}

__device__ __forceinline__ void gatherPhotons(float3 hitpos, float3 n,
    float r, float r2, int maxPhotonDepth, float3& flux, int& count)
{
    if (params.use_grid)
        gatherPhotonsGrid(hitpos, n, r, r2, maxPhotonDepth, flux, count);
    else
        gatherBruteForce(hitpos, n, r, r2, maxPhotonDepth, flux, count);
}

// Irradiance from the photon map at a surface point.
//
// The cone filter w(d) = 1 - d/r integrates to  ∫(1 - d/r) dA = π r² / 3  over
// the disc, so the density estimate is  E = 3 Σ w Φ / (π r²).
//
// With a single fixed radius the estimate is either too noisy (sparse regions)
// or too blurry (dense ones) — the "big stains".  Probing the density at the
// maximum radius first and then shrinking towards `target_photons` (photon
// count grows with r², hence the sqrt) keeps well-lit regions sharp without
// starving the dark ones.
__device__ float3 photonIrradiance(float3 hitpos, float3 n, int maxPhotonDepth,
    float& usedRadius)
{
    float r = params.gather_radius;
    float r2 = r * r;
    float3 flux;
    int count;

    gatherPhotons(hitpos, n, r, r2, maxPhotonDepth, flux, count);

    if (params.adaptive_radius && params.target_photons > 0 && count > params.target_photons) {
        float scale = sqrtf((float)params.target_photons / (float)count);
        r *= fmaxf(scale, 0.05f);
        r2 = r * r;
        gatherPhotons(hitpos, n, r, r2, maxPhotonDepth, flux, count);
    }

    usedRadius = r;
    if (count == 0)
        return make_float3(0.f, 0.f, 0.f);
    return flux * (3.f / (M_PIf * r2));
}

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

// Seed mixer. Paths are seeded from (launch index, frame index); feeding those
// in as a plain linear combination leaves neighbouring samples visibly
// correlated, so both are avalanched first.
__device__ __forceinline__ unsigned int hashSeed(unsigned int a, unsigned int b)
{
    unsigned int s = a * 0x9E3779B1u + b * 0x85EBCA6Bu + 0x165667B1u;
    s ^= s >> 16;
    s *= 0x7FEB352Du;
    s ^= s >> 15;
    s *= 0x846CA68Bu;
    s ^= s >> 16;
    return s;
}

// Every kernel draws from its own stream, and every run can be given its own
// seed.  Both matter beyond tidiness:
//
//   * the stream constants keep the path tracer and the gather off each other's
//     random numbers.  Sharing one — which is what the old
//     `pixel*1973 + frame*9277 + 4801` did in both kernels — makes the two
//     images share their noise, so the difference between them comes out
//     smaller than it should and a comparison of the two modes flatters photon
//     mapping.
//   * params.seed leaves the renderer reproducible by default (seed 0) while
//     letting two runs of the *same* settings be made genuinely independent,
//     which is what proof.py needs to measure how noisy a render is without
//     any assumption about how the runs relate.
//
// Note that the stream still depends only on (index, frame, seed), so frame i
// is the same frame in an 8-frame and a 128-frame run: short runs remain
// prefixes of long ones.
#define RNG_STREAM_PATH 0x9E3779B9u
#define RNG_STREAM_GATHER 0xBB67AE85u
#define RNG_STREAM_PHOTON 0x3C6EF372u

__device__ __forceinline__ unsigned int hashSeed3(unsigned int a, unsigned int b,
    unsigned int c)
{
    unsigned int s = a * 0x9E3779B1u + b * 0x85EBCA6Bu + c * 0xC2B2AE35u
        + 0x165667B1u;
    s ^= s >> 16;
    s *= 0x7FEB352Du;
    s ^= s >> 15;
    s *= 0x846CA68Bu;
    s ^= s >> 16;
    return s;
}

// Cosine-weighted hemisphere sample
__device__ float3 cosineSampleHemisphere(float r1, float r2)
{
    float phi = 2.f * M_PIf * r1, s = sqrtf(r2);
    return make_float3(cosf(phi) * s, sinf(phi) * s, sqrtf(1.f - r2));
}
__device__ __forceinline__ float pdfCosineHemisphere(float cosT) { return fmaxf(cosT, 0.f) * M_1_PIf; }

// Uniform direction on the sphere (pdf = 1/4π) — the emission distribution for
// environment photons.
__device__ float3 uniformSampleSphere(float r1, float r2)
{
    float z = 1.f - 2.f * r1;
    float s = sqrtf(fmaxf(0.f, 1.f - z * z));
    float phi = 2.f * M_PIf * r2;
    return make_float3(s * cosf(phi), s * sinf(phi), z);
}

// Uniform point on the unit disc (concentric mapping — no clustering at the
// centre like the naive sqrt(r),θ parameterisation).
__device__ float2 concentricSampleDisk(float r1, float r2)
{
    float ox = 2.f * r1 - 1.f, oy = 2.f * r2 - 1.f;
    if (ox == 0.f && oy == 0.f)
        return make_float2(0.f, 0.f);
    float r, theta;
    if (fabsf(ox) > fabsf(oy)) {
        r = ox;
        theta = (M_PIf * 0.25f) * (oy / ox);
    } else {
        r = oy;
        theta = (M_PIf * 0.5f) - (M_PIf * 0.25f) * (ox / oy);
    }
    return make_float2(r * cosf(theta), r * sinf(theta));
}

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

// Non-symmetric scattering and refraction.
//
// Refraction is the one interaction here that is not reciprocal.  Snell's law
// compresses the transmitted cone, and conserving the power in the beam then
// forces
//
//     L_t = L_i * (n_t / n_i)^2,
//
// so radiance jumps by n^2 on the way into a denser medium and drops by n^2 on
// the way out — L/n^2 is what is actually invariant along a ray.  Photon power
// is flux, and flux crosses the interface untouched.  Which BTDF a walk must
// use therefore depends on the quantity it carries: f_t(wi -> wo) and
// f_t(wo -> wi) differ by exactly that n^2, so a light-side walk uses the
// adjoint of the BSDF a camera-side walk uses.  pbrt 4th ed., 9.5.2
// "Non-Symmetric Scattering and Refraction".
//
// For this file it comes down to one rule, with `eta` below already being
// n_i/n_t:
//
//   * camera-side walks (__raygen__rg, __raygen__gather) carry radiance, and
//     multiply their throughput by eta*eta on every transmission;
//   * __raygen__photon carries power, and multiplies by nothing.
//
// Omitting it in all three passes looks harmless, because along a path that
// enters the glass and leaves it again the two factors are reciprocal and
// cancel.  They stop cancelling the moment a path ends inside the medium — a
// diffuse surface in contact with, or sunk into, the glass — where the camera
// walk crosses the boundary an odd number of times.  Photon mapping joins its
// two half-paths exactly at such a vertex, so it was reading n^2 = 2.25 too
// bright on the floor under the glass box, while path tracing kept both of its
// endpoints in air and never noticed.

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

// Shadow ray (ray type 1).  Only __miss__shadow runs, and it sets payload 0 to
// 1, so a payload that stays 0 means something was hit.
__device__ bool unoccluded(float3 origin, float3 dir, float tmax)
{
    if (tmax <= params.scene_epsilon)
        return false;
    unsigned int vis = 0u;
    optixTrace(params.handle, origin, dir, params.scene_epsilon, tmax, 0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0, 1, 1, vis);
    return (vis == 1u);
}

__device__ __forceinline__ bool isVisible(float3 origin, float3 target)
{
    float3 d = target - origin;
    float dist = length(d);
    if (dist <= 2.f * params.scene_epsilon)
        return false;
    return unoccluded(origin, d * (1.f / dist), dist - 2.f * params.scene_epsilon);
}

// Does `dir` escape the scene and reach the sky?
__device__ __forceinline__ bool reachesSky(float3 origin, float3 dir)
{
    return unoccluded(origin, dir, 1e16f);
}

// MIS power heuristic (β=2)
__device__ __forceinline__ float misWeight(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA, b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-10f);
}

// Direct illumination on a Lambertian surface, by next-event estimation.
//
// Photon mapping *can* deliver this term straight from the map, but a density
// estimate of the first bounce is exactly where the estimator is worst: right
// at a shadow boundary the disc straddles lit and unlit surface, which is what
// paints the soft blotches over every shadow.  Two shadow rays are far cheaper
// and far quieter, and they leave the photon map to do what it is good at —
// indirect light and caustics.
__device__ float3 directLighting(float3 x, float3 n, float3 albedo, unsigned int& rng)
{
    float3 L = make_float3(0.f, 0.f, 0.f);
    const float3 org = x + params.scene_epsilon * n;

    // Emissive triangles: pick one uniformly, then a uniform point on it.
    if (params.num_lights > 0) {
        int lIdx = min((int)(randf(rng) * params.num_lights), params.num_lights - 1);
        const EmissiveTriangle& lt = params.lights[lIdx];

        float3 lp = sampleTriangle(lt.v0, lt.v1, lt.v2, randf(rng), randf(rng));
        float3 toLight = lp - x;
        float dist2 = dot(toLight, toLight);
        float dist = sqrtf(dist2);
        float3 wi = toLight * (1.f / dist);
        float cosN = dot(n, wi);
        float cosLight = -dot(lt.normal, wi);

        if (cosN > 0.f && cosLight > 1e-4f && isVisible(org, lp)) {
            // area pdf -> solid angle pdf
            float pLightSA = (dist2 / cosLight) / (params.num_lights * lt.area);
            L += albedo * M_1_PIf * lt.emission * cosN * (1.f / pLightSA);
        }
    }

    // Environment: a cosine-weighted direction, whose pdf cancels the BRDF's
    // cosine exactly, leaving just albedo * sky.
    if (params.sky_intensity > 0.f) {
        float3 t, b;
        onb(n, t, b);
        float3 wi = normalize(toWorld(cosineSampleHemisphere(randf(rng), randf(rng)), n, t, b));
        if (reachesSky(org, wi))
            L += albedo * skyRadiance(wi, params.sky_intensity);
    }

    return L;
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

    unsigned int rng = hashSeed3(pixel, (unsigned int)params.frame_index,
        (unsigned int)params.seed ^ RNG_STREAM_PATH);

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
                params.scene_epsilon, 1e16f, 0.f,
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
                    // float3 ln = normalize(cross(lt.v1 - lt.v0, lt.v2 - lt.v0));
                    float cosLight = -dot(lt.normal, wi); // light face-forward check
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
                float3 newdir = normalize(dir - 2.f * dot(dir, n) * n);
                throughput = throughput * albedo;
                nextBsdfPdf = 0.f; // delta -> MIS uses full emission weight
                origin = hitpos + params.scene_epsilon * newdir;
                dir = newdir;

            } else {
                // Glass: stochastic Fresnel
                // using randf(rng) for Fresnel decision
                float ior = params.materials[mat_id].ior;
                // outsideFlag: CH set 1 if dot(raydir, geo_n) < 0 (entering medium)
                float eta = outsideFlag ? (1.f / ior) : ior;
                // n is already face-forward (opposite to incoming dir) from CH
                float cosT = fabsf(dot(-dir, n));
                float fr = schlick(cosT, ior);

                float3 newdir;
                bool doReflect = !refractDir(dir, n, eta, newdir); // TIR
                if (!doReflect)
                    doReflect = (randf(rng) < fr); // stochastic Fresnel
                if (doReflect) {
                    newdir = normalize(dir - 2.f * dot(dir, n) * n);
                } else {
                    // This walk carries radiance, so a transmission scales it
                    // by eta*eta; see the note above refractDir.
                    throughput = throughput * (eta * eta);
                }

                nextBsdfPdf = 0.f;
                origin = hitpos + params.scene_epsilon * newdir;
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
    const unsigned int photon_id = optixGetLaunchIndex().x;
    unsigned int rng = hashSeed3(photon_id, (unsigned int)params.frame_index,
        (unsigned int)params.seed ^ RNG_STREAM_PHOTON);

    const bool hasEnv = (params.emit_sky_photons != 0)
        && (params.sky_intensity > 0.f)
        && (params.scene_radius > 0.f);
    const bool hasLights = (params.num_lights > 0);
    if (!hasEnv && !hasLights)
        return;

    // Split the paths between the two emitters in proportion to their flux
    // (the host works the ratio out once per pass).
    const float pEnv = hasLights ? (hasEnv ? params.sky_select_prob : 0.f) : 1.f;

    float3 origin, dir, power;

    if (randf(rng) < pEnv) {
        // Environment light.  Choose a travel direction uniformly on the
        // sphere, then a start point on the disc of radius R facing it, pushed
        // back so the photon enters from outside the scene:
        //     Φ = L(-ω) · πR² / p(ω),   p(ω) = 1/4π   ->   Φ = L(-ω) · 4π²R²
        float3 w = uniformSampleSphere(randf(rng), randf(rng));
        float3 t, b;
        onb(w, t, b);
        float2 d = concentricSampleDisk(randf(rng), randf(rng));
        const float R = params.scene_radius;

        origin = params.scene_center + R * (d.x * t + d.y * b) - R * w;
        dir = w;
        power = skyRadiance(-w, params.sky_intensity)
            * ((4.f * M_PIf * M_PIf * R * R)
                / ((float)params.num_photon_paths * pEnv));
    } else {
        // Emissive triangle: uniform over the list, uniform over its area,
        // cosine-weighted over its hemisphere.  A Lambertian emitter radiates
        // Φ = L_e · A · π.
        int lIdx = min((int)(randf(rng) * params.num_lights), params.num_lights - 1);
        const EmissiveTriangle& lt = params.lights[lIdx];

        float3 ln = lt.normal; // precomputed face normal
        float3 t, b;
        onb(ln, t, b);
        float3 local = cosineSampleHemisphere(randf(rng), randf(rng));

        origin = sampleTriangle(lt.v0, lt.v1, lt.v2, randf(rng), randf(rng))
            + params.scene_epsilon * ln;
        dir = normalize(toWorld(local, ln, t, b));
        power = lt.emission
            * ((lt.area * M_PIf * (float)params.num_lights)
                / ((float)params.num_photon_paths * (1.f - pEnv)));
    }

    power = power * params.photon_power_scale;

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
            params.scene_epsilon, 1e16f, 0.f,
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

            // depth == 0 is a photon straight off the light, i.e. the direct
            // term.  The gather computes that analytically with shadow rays,
            // so storing it here would both double-count and hand the noisiest
            // part of the image to the density estimate.  Everything past the
            // first bounce — indirect light and caustics through L S+ D — is
            // what the map is for.
            if (depth > 0 || params.store_direct_photons) {
                int slot = atomicAdd(params.photon_count, 1);
                if (slot < params.photon_capacity) {
                    Photon ph;
                    ph.pos = hitpos;
                    ph.power = power;
                    ph.dir = -dir;
                    ph.normal = n;
                    ph.depth = depth;
                    params.photon_map[slot] = ph;
                }
            }

            if (randf(rng) > survive)
                break;
            power = power * albedo * (1.f / survive);

            float3 tv, bv;
            onb(n, tv, bv);
            float3 loc = cosineSampleHemisphere(randf(rng), randf(rng));
            dir = normalize(toWorld(loc, n, tv, bv));
            origin = hitpos + params.scene_epsilon * n;

        } else if (matType == MAT_MIRROR) {
            dir = normalize(dir - 2.f * dot(dir, n) * n);
            power = power * albedo;
            origin = hitpos + params.scene_epsilon * dir;

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
            origin = hitpos + params.scene_epsilon * dir;
        }
    }
}

extern "C" __global__ void __raygen__gather()
{
    const uint3 idx = optixGetLaunchIndex();
    const int pixel = idx.y * params.width + idx.x;
    unsigned int rng = hashSeed3(pixel, (unsigned int)params.frame_index,
        (unsigned int)params.seed ^ RNG_STREAM_GATHER);

    float3 result_over_samples = make_float3(0.f, 0.f, 0.f);

    for (int s = 0; s < params.samples_per_pixel; s++) {
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
                params.scene_epsilon, 1e16f, 0.f,
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
                // Direct light by next-event estimation.  Skipped when the map
                // was told to carry the direct photons itself, otherwise the
                // two would double-count each other.
                if (!params.store_direct_photons)
                    result += path_throughput * directLighting(hitpos, n, albedo, rng);

                // Indirect light (and caustics) from the photon map.
                //
                // The camera walk has spent `depth + 1` vertices to get here
                // (specular bounces included), and a photon stored at its own
                // bounce j carries j + 1 vertices of which the last one is this
                // very point.  The finished path therefore has j + depth + 1
                // vertices, and the path tracer would have allowed at most
                // max_depth of them — so a photon may contribute only while
                //
                //     j + depth + 1 <= max_depth.
                //
                // At depth 0 that admits photons up to j = max_depth-1, exactly
                // matching a path-traced v1..v_D; at depth = max_depth-1 it
                // admits none, matching a path tracer left with only its final
                // shadow ray.  Without this the two budgets are independent and
                // photon mapping quietly integrates far longer paths.
                int maxPhotonDepth = params.max_depth - 1 - depth;
                float usedRadius;
                float3 irradiance = photonIrradiance(hitpos, n, maxPhotonDepth,
                    usedRadius);

                // Lambertian BRDF
                result += path_throughput * albedo * irradiance * M_1_PIf;
                break; // PM: only first diffuse bounce

            } else if (matType == MAT_MIRROR) {
                path_throughput = path_throughput * albedo;
                dir = normalize(dir - 2.f * dot(dir, n) * n);
                origin = hitpos + params.scene_epsilon * dir;

            } else { // MAT_GLASS
                float ior = params.materials[mat_id].ior;
                float eta = outsideFlag ? (1.f / ior) : ior;
                float cosT = fabsf(dot(-dir, n));
                float fr = schlick(cosT, ior);
                float3 newdir;
                bool doRefl = !refractDir(dir, n, eta, newdir);
                if (!doRefl)
                    doRefl = (randf(rng) < fr);
                if (doRefl) {
                    newdir = normalize(dir - 2.f * dot(dir, n) * n);
                } else {
                    // The gather walk carries radiance just like the path
                    // tracer's, and this is the half of the photon-mapped path
                    // that crosses the boundary an odd number of times when the
                    // photon sits inside the glass.  See the note above
                    // refractDir.
                    path_throughput = path_throughput * (eta * eta);
                }
                dir = newdir;
                origin = hitpos + params.scene_epsilon * dir;
            }
        }

        result_over_samples += result;
    }

    result_over_samples.x /= params.samples_per_pixel;
    result_over_samples.y /= params.samples_per_pixel;
    result_over_samples.z /= params.samples_per_pixel;

    // Temporal accumulation
    float3 prev = (params.frame_index == 0)
        ? make_float3(0.f, 0.f, 0.f)
        : params.accum_buffer[pixel];
    float3 acc = make_float3(prev.x + result_over_samples.x,
        prev.y + result_over_samples.y,
        prev.z + result_over_samples.z);
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
        albedo = sampleTexSRGB(mat.base_color_tex, uvu, uvv, mat.albedo);
        albedo = albedo * mat.albedo;
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

    if (!outside) {
        emission = make_float3(0.f, 0.f, 0.f); // no emission from backface
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
                    float cosL = dot(params.lights[li].normal, -raydir);
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
    // The sky lives in params (not in the miss record) so that the photon
    // emitter can shoot the very same environment — see skyRadiance().
    float3 raydir = optixGetWorldRayDirection();
    float3 sky = skyRadiance(raydir, params.sky_intensity);

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

// Photon visualisation: splat every stored photon straight onto the film.
//
// A photon carries *flux* (Φ, in watts), but a pixel has to end up holding
// *radiance*.  Adding Φ to a pixel — as this used to do — measures the wrong
// quantity entirely, and because flux does not depend on where you stand, every
// photon stayed equally bright from every distance and every angle.  The
// missing piece is the area of surface that one pixel actually covers:
//
//   pixel area on the image plane   A_pix = (2|cam_u|/W) · (2|cam_v|/H)
//   solid angle it subtends         dω    = A_pix · cos³θ / f²
//   surface patch it sees           dA    = dω · dist² / cosSurf
//
// with f = |cam_w| the image-plane distance, θ the angle off the camera axis
// (cosθ = depth/dist) and cosSurf the angle between the photon's surface normal
// and the direction back to the eye.  cos³θ is the usual pinhole falloff: one
// step of cosine for the foreshortening of the pixel itself and two for it
// being further away off-axis.
//
// Irradiance is then E = Φ/dA, and a white Lambertian surface reflects
// L = E/π.  That is what gets splatted, so the result is directly comparable to
// the photon-mapped image: tilt a surface away and dA grows, each photon dims,
// but more of them land in the pixel — the two cancel, exactly as they should
// for a diffuse surface, instead of the flat over-bright wash from before.
extern "C" __global__ void __raygen__lightvis()
{
    const int pid = (int)optixGetLaunchIndex().x;

    const int stored = min(*params.photon_count, params.photon_capacity);
    if (pid >= stored)
        return;

    const Photon& ph = params.photon_map[pid];

    // Camera frame.  cam_w reaches the centre of the image plane, so its length
    // is the plane distance; cam_u / cam_v are the plane's half-extents.
    const float f = length(params.cam_w);
    const float u_scale = length(params.cam_u);
    const float v_scale = length(params.cam_v);
    if (f < 1e-10f || u_scale < 1e-10f || v_scale < 1e-10f)
        return;

    const float3 fwd = params.cam_w * (1.f / f);
    const float3 right = params.cam_u * (1.f / u_scale);
    const float3 up = params.cam_v * (1.f / v_scale);

    const float3 toP = ph.pos - params.cam_eye;
    const float depth = dot(toP, fwd);
    if (depth < params.scene_epsilon)
        return; // behind the camera

    const float dist2 = dot(toP, toP);
    const float dist = sqrtf(dist2);
    const float3 view = toP * (1.f / dist); // eye -> photon

    // Only photons on a surface turned towards the camera are visible.
    const float cosSurf = dot(ph.normal, -view);
    if (cosSurf < 1e-4f)
        return;

    // NDC in [-1,1], matching the primary ray construction in __raygen__rg.
    const float px = dot(toP, right) * f / (depth * u_scale);
    const float py = dot(toP, up) * f / (depth * v_scale);
    if (px < -1.f || px > 1.f || py < -1.f || py > 1.f)
        return;

    if (!isVisible(ph.pos + params.scene_epsilon * ph.normal, params.cam_eye))
        return;

    // Flux -> radiance.
    const float A_pix = (2.f * u_scale / (float)params.width)
        * (2.f * v_scale / (float)params.height);
    const float cosT = depth / dist;
    const float dA = A_pix * cosT * cosT * cosT * dist2 / (f * f * cosSurf);
    if (dA < 1e-20f)
        return;

    const float3 L = ph.power * (M_1_PIf / dA);

    // Continuous film coordinates (pixel centres sit at integer + 0.5).
    const float fx = (px * 0.5f + 0.5f) * (float)params.width - 0.5f;
    const float fy = (py * 0.5f + 0.5f) * (float)params.height - 0.5f;
    const int cx = (int)floorf(fx + 0.5f);
    const int cy = (int)floorf(fy + 0.5f);

    // A one-pixel splat of a point sample is extremely speckly, so spread the
    // photon over a small separable tent whose weights sum to one.  That
    // conserves flux — it only blurs the estimate, it does not brighten it.
    const int R = min(max(params.lightvis_splat_px, 0), 4);
    float wx[9], wy[9];
    float sx = 0.f, sy = 0.f;
    for (int i = -R; i <= R; ++i) {
        wx[i + R] = fmaxf(1.f - fabsf((float)(cx + i) - fx) / (float)(R + 1), 0.f);
        wy[i + R] = fmaxf(1.f - fabsf((float)(cy + i) - fy) / (float)(R + 1), 0.f);
        sx += wx[i + R];
        sy += wy[i + R];
    }
    if (sx <= 0.f || sy <= 0.f)
        return;

    // lightvis_buffer is float* with 3 interleaved floats per pixel
    float* buf = params.lightvis_buffer;
    const int W = (int)params.width, H = (int)params.height;

    for (int j = -R; j <= R; ++j) {
        int y = cy + j;
        if (y < 0 || y >= H)
            continue;
        for (int i = -R; i <= R; ++i) {
            int x = cx + i;
            if (x < 0 || x >= W)
                continue;
            float w = (wx[i + R] / sx) * (wy[j + R] / sy);
            if (w <= 0.f)
                continue;
            int pixel = y * W + x;
            atomicAdd(buf + pixel * 3 + 0, L.x * w);
            atomicAdd(buf + pixel * 3 + 1, L.y * w);
            atomicAdd(buf + pixel * 3 + 2, L.z * w);
        }
    }
}