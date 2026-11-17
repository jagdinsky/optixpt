// The OptiX programs of the adaptive LoD hash-grid SPPM (sppm_lepski.h):
// stage 1, the camera walk, and stage 2, the photon trace.
//
// device.cu is included whole, and not one line of it is changed.  It is
// where the renderer's conventions live — the camera ray, the payload, next
// event estimation, the emitters, the materials — and using the very same
// source is what keeps this method converging to the image the path tracer
// and the other photon mappers converge to.  The price is that this module
// carries a copy of every existing program; none of them is linked into this
// method's pipelines, and the existing pipelines are built from device.ptx
// as before, so neither side can move the other's code or registers.
#include "device.cu"

#include "sppm_lepski.h"

// The storage roulette draws from a stream of its own, so that the photon
// paths are the ones every other method traces for the same frame and seed.
#define RNG_STREAM_LEPSKI_RR 0x1F83D9ABu

__device__ __forceinline__ const LepskiParams& lepskiParams()
{
    return *reinterpret_cast<const LepskiParams*>(optixGetSbtDataPointer());
}

// Stage 1 for one pixel.  This is cameraWalk (device.cu) — the same jitter,
// the same random stream, the same specular chain, emission and next event
// estimation, hence the same vertex and the same direct light as the other
// photon mappers' camera pass — plus what this method needs at the vertex:
// the direction back along the path and the gather radius.
__device__ __forceinline__ LepskiGatherPoint lepskiCameraWalk(uint3 idx, int pixel,
    const LepskiParams& L)
{
    unsigned int rng = hashSeed3(pixel, (unsigned int)params.frame_index,
        (unsigned int)params.seed ^ RNG_STREAM_GATHER);

    LepskiGatherPoint gp;
    gp.pos = make_float3(0.f, 0.f, 0.f);
    gp.r_base = 0.f;
    gp.normal = make_float3(0.f, 0.f, 1.f);
    gp.t = 0.f;
    gp.weight = make_float3(0.f, 0.f, 0.f);
    gp.info = 0;
    gp.direct = make_float3(0.f, 0.f, 0.f);
    gp.wo_oct = 0u;

    float pu = (idx.x + randf(rng)) / params.width;
    float pv = (idx.y + randf(rng)) / params.height;

    float3 origin = params.cam_eye;
    float3 dir = normalize(params.cam_w
        + (2.f * pu - 1.f) * params.cam_u
        + (2.f * pv - 1.f) * params.cam_v);

    float3 result = make_float3(0.f, 0.f, 0.f);
    float3 path_throughput = make_float3(1.f, 1.f, 1.f);
    float pathLength = 0.f; // eye to the current vertex, along the path

    for (int depth = 0; depth < params.max_depth; depth++) {
        // The payload of device.cu's hit and miss programs.
        unsigned int p0 = 0u, p1 = 0u, p2 = 0u;
        unsigned int p3 = __float_as_uint(1.f);
        unsigned int p4 = __float_as_uint(1.f);
        unsigned int p5 = __float_as_uint(1.f);
        unsigned int p6 = 0u; // no BSDF pdf: an emitter hit counts in full
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
            // the sky
            result += path_throughput * make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
            break;
        }

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

        pathLength += length(hitpos - origin);

        // an emitter seen directly or through the specular chain
        float3 emission = make_float3(__uint_as_float(p0),
            __uint_as_float(p1),
            __uint_as_float(p2));
        result += path_throughput * emission;

        if (matType == MAT_DIFFUSE) {
            // Direct light by next event estimation, always: this method
            // never stores a photon's first hit.
            result += path_throughput * directLighting(hitpos, n, albedo, rng);

            gp.pos = hitpos;
            gp.normal = n;
            gp.weight = path_throughput * albedo * M_1_PIf;
            gp.wo_oct = PackOctahedral(-dir);

            // The path-length budget the camera walk and the photon share;
            // see cameraWalk.  A vertex left with none of it gathers nothing.
            const int maxPhotonDepth = params.max_depth - 1 - depth;
            gp.t = pathLength;
            if (maxPhotonDepth >= 1) {
                int lMin, lMax;
                gp.r_base = lepskiRadius(L, pathLength, params.pixel_angle,
                    fabsf(dot(n, dir)), length(hitpos - params.cam_eye), params.radius_min,
                    lMin, lMax);
                gp.info = lepskiPackInfo(lMin, lMax, maxPhotonDepth);
            }
            break;

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
                // radiance scales by eta^2 across the boundary (device.cu,
                // above refractDir)
                path_throughput = path_throughput * (eta * eta);
            }
            dir = newdir;
            origin = hitpos + params.scene_epsilon * dir;
        }
    }

    gp.direct = result;
    return gp;
}

extern "C" __global__ void __raygen__lepski_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const int pixel = idx.y * params.width + idx.x;
    const LepskiParams& L = lepskiParams();

    const LepskiGatherPoint gp = lepskiCameraWalk(idx, pixel, L);
    L.gather_points[pixel] = gp;

    // The buckets this pixel's gather will search are the only ones whose
    // photons it can use.  Thousands of pixels share a bucket, so the bit is
    // read first and set only by the few threads that find it clear.
    if (gp.r_base > 0.f && L.culling) {
        unsigned int buckets[16];
        const int count = lepskiGatherBuckets(L, gp.pos, lepskiLevelMin(gp.info),
            lepskiLevelMax(gp.info), buckets);
        for (int i = 0; i < count; ++i) {
            const unsigned int bit = 1u << (buckets[i] & 31u);
            unsigned int* word = &L.active_bits[buckets[i] >> 5];
            if (!(*word & bit))
                atomicOr(word, bit);
        }
    }
}

// The next free photon slot, one atomic per warp instead of one per photon:
// of the threads that reach this point together, the first reserves a slot
// for each and the others take theirs by rank (Adinets, "Optimized filtering
// with warp-aggregated atomics").  The result does not depend on which
// threads share a warp, which is the condition under which OptiX allows
// warp-wide instructions (Programming Guide, "Programming model"); the lane
// mask is read and used with no trace call in between.
#ifndef LEPSKI_WARP_AGGREGATE
#define LEPSKI_WARP_AGGREGATE 1
#endif
__device__ __forceinline__ unsigned int lepskiAppendSlot(unsigned int* counter)
{
#if LEPSKI_WARP_AGGREGATE
    const unsigned int active = __activemask();
    unsigned int lower;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lower));
    const unsigned int rank = __popc(active & lower);
    unsigned int base = 0u;
    if (rank == 0u)
        base = atomicAdd(counter, (unsigned int)__popc(active));
    base = __shfl_sync(active, base, __ffs(active) - 1);
    return base + rank;
#else
    return atomicAdd(counter, 1u);
#endif
}

// Stage 2 for one photon path: tracePhotonPath's emission and walk (device.cu)
// with the same random stream, so the paths are the ones the other methods
// trace, and this method's own deposit.
extern "C" __global__ void __raygen__lepski_photon()
{
    const LepskiParams& L = lepskiParams();
    const unsigned int photon_id = optixGetLaunchIndex().x;
    unsigned int rng = hashSeed3(photon_id, (unsigned int)params.frame_index,
        (unsigned int)params.seed ^ RNG_STREAM_PHOTON);

    const bool hasEnv = (params.emit_sky_photons != 0)
        && (params.sky_intensity > 0.f)
        && (params.scene_radius > 0.f);
    const bool hasLights = (params.num_lights > 0);
    if (!hasEnv && !hasLights)
        return;

    const float pEnv = hasLights ? (hasEnv ? params.sky_select_prob : 0.f) : 1.f;

    float3 origin, dir, power;

    if (randf(rng) < pEnv) {
        // Environment light: a direction uniform on the sphere, a start point
        // on the disc of the scene's bounding sphere facing it.
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
        // Emissive triangle: uniform over the list and its area, cosine-
        // weighted over its hemisphere.
        int lIdx = min((int)(randf(rng) * params.num_lights), params.num_lights - 1);
        const EmissiveTriangle& lt = params.lights[lIdx];

        float3 ln = lt.normal;
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

    unsigned int candidates = 0u; // deposits past the first hit
    unsigned int activeHits = 0u; // ... of which in a cell some gather searches

    for (int depth = 0; depth < params.max_depth; depth++) {
        unsigned int p0 = 0u, p1 = 0u, p2 = 0u;
        unsigned int p3 = __float_as_uint(1.f);
        unsigned int p4 = __float_as_uint(1.f);
        unsigned int p5 = __float_as_uint(1.f);
        unsigned int p6 = 0u;
        unsigned int p7 = 0u, p8 = 0u, p9 = 0u;
        unsigned int p10 = 0u, p11 = 0u, p12 = 0u;
        unsigned int p13 = 0u;
        unsigned int p14 = 0u;
        unsigned int p15 = 0u;

        optixTrace(
            params.handle, origin, dir,
            params.scene_epsilon, 1e16f, 0.f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1, p2, p3, p4, p5, p6,
            p7, p8, p9, p10, p11, p12,
            p13, p14, p15);

        if (p15 == 1u)
            break;

        unsigned int matType = p13 & 0xFFu;
        unsigned int mat_id = p14;
        bool outsideFlag = (p13 >> 8u) & 1u;

        float3 hitpos = make_float3(__uint_as_float(p7),
            __uint_as_float(p8),
            __uint_as_float(p9));
        float3 n = make_float3(__uint_as_float(p10),
            __uint_as_float(p11),
            __uint_as_float(p12));
        float3 albedo = make_float3(__uint_as_float(p3),
            __uint_as_float(p4),
            __uint_as_float(p5));

        if (matType == MAT_DIFFUSE) {
            float survive = fmaxf(albedo.x, fmaxf(albedo.y, albedo.z));
            survive = fmaxf(survive, 0.05f);

            // The first hit is direct light, which stage 1 settles with
            // shadow rays; storing it too would count it twice.
            if (depth > 0) {
                ++candidates;
                // Stored only where some gather of this iteration looks.  The
                // path itself goes on either way: it can still bounce from
                // here into a cell that is searched.
                const unsigned int bucket = lepskiPhotonBucket(L, hitpos, params.cam_eye);
                const bool active = !L.culling
                    || ((L.active_bits[bucket >> 5] >> (bucket & 31u)) & 1u) != 0u;
                if (active) {
                    ++activeHits;

                    // A weak photon is stored with probability q and 1/q of
                    // its flux — the same in expectation, with fewer photons
                    // to sort and test.  The path keeps its own flux.  The
                    // random number belongs to this (path, bounce) alone, so
                    // the decision is the same whatever else was or was not
                    // stored.
                    float3 flux = power;
                    bool store = true;
                    const float lum = lepskiLuminance(power);
                    if (lum < L.flux_threshold) {
                        const float q = fmaxf(0.05f, lum / L.flux_threshold);
                        unsigned int rr = hashSeed3(photon_id, (unsigned int)params.frame_index,
                            ((unsigned int)params.seed ^ RNG_STREAM_LEPSKI_RR)
                                + (unsigned int)depth * 0x9E3779B1u);
                        store = lum > 0.f && randf(rr) < q;
                        flux = power * (1.f / q);
                    }

                    if (store) {
                        const unsigned int slot = lepskiAppendSlot(&L.counters[LEPSKI_COUNT_STORED]);
                        if (slot < L.max_stored) {
                            PackedPhoton ph;
                            ph.pos = hitpos;
                            ph.wiOct = PackOctahedral(-dir);
                            ph.flux = flux;
                            ph.nOct = packNormalDepth(n, depth);
                            L.raw_photons[slot] = ph;
                            L.raw_hashes[slot] = bucket;
                        }
                    }
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

    // the culling statistics: one atomic per path, not per deposit
    if (candidates > 0u) {
        atomicAdd(&L.counters[LEPSKI_COUNT_CANDIDATES], candidates);
        atomicAdd(&L.counters[LEPSKI_COUNT_ACTIVE], activeHits);
    }
}
