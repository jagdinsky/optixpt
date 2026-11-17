// The OptiX programs of RTPM (rtpm.h): the photon generation pass, and the
// collection pass with its intersection and any-hit programs.
//
// device.cu is included whole, and not one line of it is changed: it is where
// the renderer's conventions live, and the photon pass below walks exactly
// the paths its tracePhotonPath walks.  This module is compiled for pipelines
// that also hold custom primitives (the photon boxes), which the renderer's
// own module is not, so its pipelines take their miss and hit programs from
// the copies in here; the existing pipelines are built from device.ptx as
// before.
#include "device.cu"

#include "rtpm.h"

// The rejection step and Stochastic Evaluation draw from streams of their
// own, so that the photon paths are the ones every other method traces for
// the same frame and seed.
#define RNG_STREAM_RTPM_REJECT 0x5BE0CD19u
#define RNG_STREAM_RTPM_LIST 0xCBBB9D5Du

__device__ __forceinline__ const RtpmParams& rtpmParams()
{
    return *reinterpret_cast<const RtpmParams*>(optixGetSbtDataPointer());
}

// --- Photon generation (paper secs. 3.4, 3.5) -------------------------------
//
// tracePhotonPath's emission and walk (device.cu), with the same random
// stream.  What differs is the deposit: the photon goes into the caustic or
// the global map, as a box for the structure and a record for the shading.
extern "C" __global__ void __raygen__rtpm_photon()
{
    const RtpmParams& R = rtpmParams();
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
        float3 w = uniformSampleSphere(randf(rng), randf(rng));
        float3 t, b;
        onb(w, t, b);
        float2 d = concentricSampleDisk(randf(rng), randf(rng));
        const float Rs = params.scene_radius;

        origin = params.scene_center + Rs * (d.x * t + d.y * b) - Rs * w;
        dir = w;
        power = skyRadiance(-w, params.sky_intensity)
            * ((4.f * M_PIf * M_PIf * Rs * Rs)
                / ((float)params.num_photon_paths * pEnv));
    } else {
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

    // Until its first diffuse bounce a path has come from the light through
    // specular surfaces only: what it deposits then is a caustic photon.
    bool specularOnly = true;
    unsigned int candidates = 0u; // deposits past the first hit
    unsigned int visible = 0u; // ... of which in a cell some gather is near

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

            // The first hit is direct light, which the camera pass settles
            // with shadow rays; storing it too would count it twice.
            if (depth > 0) {
                ++candidates;

                // Photon Culling: stored only if the cell it lies in was
                // marked by a gather point.  Cells that share a mask entry
                // let photons through that nobody collects — that is all a
                // collision costs.  The path goes on either way.
                bool marked = true;
                if (R.culling) {
                    const unsigned int cell = rtpmCullIndex(
                        photonCellCoord(hitpos.x, R.origin.x, R.cull_cell),
                        photonCellCoord(hitpos.y, R.origin.y, R.cull_cell),
                        photonCellCoord(hitpos.z, R.origin.z, R.cull_cell), R.cull_bits);
                    marked = R.cull_mask[cell] != 0;
                }
                if (marked) {
                    ++visible;
                    const int m = specularOnly ? RTPM_CAUSTIC : RTPM_GLOBAL;

                    // The rejection step: a global photon is stored with
                    // probability 1 - reject and carries 1 / (1 - reject) of
                    // its flux; caustic photons are always stored.  The path
                    // keeps its own flux.  The random number belongs to this
                    // (path, bounce) alone.
                    float3 flux = power;
                    bool store = true;
                    if (m == RTPM_GLOBAL && R.reject > 0.f) {
                        unsigned int rr = hashSeed3(photon_id, (unsigned int)params.frame_index,
                            ((unsigned int)params.seed ^ RNG_STREAM_RTPM_REJECT)
                                + (unsigned int)depth * 0x9E3779B1u);
                        store = randf(rr) >= R.reject;
                        flux = power * (1.f / (1.f - R.reject));
                    }

                    if (store) {
                        const RtpmMap& map = R.maps[m];
                        const unsigned int slot = atomicAdd(&R.counters[m], 1u);
                        if (slot < map.capacity) {
                            const float r = map.box_radius;
                            OptixAabb box;
                            box.minX = hitpos.x - r;
                            box.minY = hitpos.y - r;
                            box.minZ = hitpos.z - r;
                            box.maxX = hitpos.x + r;
                            box.maxY = hitpos.y + r;
                            box.maxZ = hitpos.z + r;
                            map.aabbs[slot] = box;

                            PackedPhoton ph;
                            ph.pos = hitpos;
                            ph.wiOct = PackOctahedral(-dir);
                            ph.flux = flux;
                            ph.nOct = packNormalDepth(n, depth);
                            map.photons[slot] = ph;
                        }
                    }
                }
            }
            specularOnly = false;

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
        atomicAdd(&R.counters[RTPM_COUNT_CANDIDATES], candidates);
        atomicAdd(&R.counters[RTPM_COUNT_VISIBLE], visible);
    }
}

// --- Photon collection (paper sec. 3.6) -------------------------------------

// Called for every box the collection ray's traversal reaches.  The ray is
// infinitely short, so those are the boxes that hold its origin, the gather
// point; the photon counts if the point is within the gather's radius of it.
// The hit is reported at the far end of the ray, the only distance there is.
extern "C" __global__ void __intersection__rtpm()
{
    const RtpmMapRecord& map = *reinterpret_cast<const RtpmMapRecord*>(optixGetSbtDataPointer());
    const PackedPhoton& ph = map.photons[optixGetPrimitiveIndex()];

    // --stats: the boxes the traversal reached
    if (map.stats)
        optixSetPayload_13(optixGetPayload_13() + 1u);

    // photonUsable's own distance, in its arithmetic
    const float3 diff = subRn(ph.pos, optixGetWorldRayOrigin());
    const float d2 = dotRn(diff, diff);
    if (d2 > __uint_as_float(optixGetPayload_7()))
        return;
    optixReportIntersection(optixGetRayTmax(), 0u, __float_as_uint(d2));
}

// Called once for every photon the intersection program reported (the boxes
// are built with REQUIRE_SINGLE_ANYHIT_CALL).  It applies the remaining tests
// and adds the photon to the sum — or, with Stochastic Evaluation, only puts
// its index in the list — and then turns the hit down, so that the traversal
// goes on to the other boxes.
extern "C" __global__ void __anyhit__rtpm()
{
    const RtpmMapRecord& map = *reinterpret_cast<const RtpmMapRecord*>(optixGetSbtDataPointer());
    const unsigned int id = optixGetPrimitiveIndex();
    const PackedPhoton& ph = map.photons[id];

    const float3 n = make_float3(__uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()), __uint_as_float(optixGetPayload_6()));
    const float3 diff = subRn(ph.pos, optixGetWorldRayOrigin());

    if (rtpmUsable(ph, diff, n, __uint_as_float(optixGetPayload_14()), params.photon_normal_tol,
            (int)optixGetPayload_8())) {
        const unsigned int accepted = optixGetPayload_3() + 1u;
        optixSetPayload_3(accepted);

        if (map.list_size == 0) {
            const float w = rtpmWeight(__uint_as_float(optixGetAttribute_0()),
                __uint_as_float(optixGetPayload_15()), map.uniform_kernel);
            float3 flux = make_float3(__uint_as_float(optixGetPayload_0()),
                __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()));
            addWeighted(flux, w, ph.flux);
            optixSetPayload_0(__float_as_uint(flux.x));
            optixSetPayload_1(__float_as_uint(flux.y));
            optixSetPayload_2(__float_as_uint(flux.z));
        } else {
            // Reservoir sampling (paper listing 4): the first list_size
            // photons fill the list; the n-th after that replaces a random
            // entry with probability list_size / n, which leaves every
            // photon equally likely to be in it.
            unsigned int slot = accepted - 1u;
            if (slot >= (unsigned int)map.list_size) {
                unsigned int rng = optixGetPayload_12();
                slot = (unsigned int)(randf(rng) * (float)accepted);
                optixSetPayload_12(rng);
            }
            if (slot == 0u)
                optixSetPayload_9(id);
            else if (slot == 1u && map.list_size > 1)
                optixSetPayload_10(id);
            else if (slot == 2u && map.list_size > 2)
                optixSetPayload_11(id);
        }
    }
    optixIgnoreIntersection();
}

// The collection ray hits nothing by design: every hit is turned down.
extern "C" __global__ void __miss__rtpm()
{
}

// One gather with one photon map: the kernel-weighted flux within r of the
// gather point, and what the traversal looked at.
__device__ __forceinline__ float3 rtpmCollectMap(const RtpmParams& R, int m, const GatherPoint& gp,
    float r, int pixel, unsigned int& tested, unsigned int& accepted)
{
    const RtpmMap& map = R.maps[m];
    const float invR = 1.f / r;

    unsigned int p0 = 0u, p1 = 0u, p2 = 0u; // flux
    unsigned int p3 = 0u; // accepted
    unsigned int p4 = __float_as_uint(gp.normal.x);
    unsigned int p5 = __float_as_uint(gp.normal.y);
    unsigned int p6 = __float_as_uint(gp.normal.z);
    unsigned int p7 = __float_as_uint(r * r);
    unsigned int p8 = (unsigned int)gp.max_photon_depth;
    unsigned int p9 = 0u, p10 = 0u, p11 = 0u; // the list
    unsigned int p12 = hashSeed3((unsigned int)pixel, (unsigned int)params.frame_index,
        ((unsigned int)params.seed ^ RNG_STREAM_RTPM_LIST) + (unsigned int)m);
    unsigned int p13 = 0u; // tested
    unsigned int p14 = __float_as_uint(params.photon_plane_tol * r);
    unsigned int p15 = __float_as_uint(invR);

    // The infinitely short ray: it starts at the gather point and ends a
    // hair further on.  Its direction means nothing; the normal is at hand.
    // SBT offset m picks the map's hit record; no closest-hit program runs.
    optixTrace(map.handle, gp.pos, gp.normal,
        0.f, params.scene_epsilon * 1e-3f, 0.f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        m, 1, 0,
        p0, p1, p2, p3, p4, p5, p6,
        p7, p8, p9, p10, p11, p12,
        p13, p14, p15);

    tested += p13;
    accepted += p3;

    float3 flux = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    if (R.list_size > 0 && p3 > 0u) {
        // Stochastic Evaluation: only the listed photons are evaluated, each
        // standing for n / min(n, list size) of the n that were found.
        const unsigned int n = p3;
        const unsigned int listed = min(n, (unsigned int)R.list_size);
        const unsigned int ids[RTPM_MAX_LIST] = { p9, p10, p11 };
        for (unsigned int i = 0; i < listed; ++i) {
            const PackedPhoton& ph = map.photons[ids[i]];
            const float3 diff = subRn(ph.pos, gp.pos);
            addWeighted(flux, rtpmWeight(dotRn(diff, diff), invR, R.uniform_kernel), ph.flux);
        }
        flux = flux * ((float)n / (float)listed);
    }
    return flux;
}

// The collection pass for one pixel: the gather point the camera pass left
// (params.gather_points), the photons of both maps around it, the pixel's
// radiance (params.query_radiance) — what the gathers' query pass does, with
// the photon structures in place of a grid.
extern "C" __global__ void __raygen__rtpm_collect()
{
    const uint3 idx = optixGetLaunchIndex();
    const int pixel = idx.y * params.width + idx.x;
    const RtpmParams& R = rtpmParams();

    const GatherPoint gp = params.gather_points[pixel];
    float3 radiance = gp.direct;
    unsigned int tested = 0u, accepted = 0u;

    if (gp.valid) {
        // The density estimate of each map with its own radius.
        float3 irradiance = make_float3(0.f, 0.f, 0.f);
        for (int m = 0; m < RTPM_MAPS; ++m) {
            if (R.maps[m].stored == 0u)
                continue;
            const float r = gp.r * R.maps[m].radius_scale;
            const float3 flux = rtpmCollectMap(R, m, gp, r, pixel, tested, accepted);
            irradiance += flux * rtpmNormalisation(r, R.uniform_kernel);
        }
        radiance += gp.weight * irradiance;
    }

    params.query_radiance[pixel] = radiance;

    // --stats: one slot per pixel; -1 = no gather here.
    if (R.stats) {
        params.stat_tested[pixel] = gp.valid ? (int)tested : -1;
        params.stat_accepted[pixel] = gp.valid ? (int)accepted : -1;
        params.stat_cells[pixel] = gp.valid ? 0 : -1;
    }
}
