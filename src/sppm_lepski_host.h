#pragma once
// The host side of the adaptive LoD hash-grid SPPM (sppm_lepski.h): its
// OptiX module and pipelines, and one iteration's four stages.
//
// Part of main.cpp's translation unit — included there once, after the
// renderer state, the pipeline helpers, the timers and the image comparison
// it uses — so that the method's host code is all in this file and main.cpp
// only registers it.
#include "sppm_lepski.h"

// A path stores at most one photon per bounce after its first, which bounds
// the map; beyond this many slots (68 bytes each) the bound is not allocated
// in full and a pass that outgrows the buffer is reported instead.
static constexpr size_t kLepskiMaxPhotonSlots = 16'000'000;

struct LepskiHost {
    // The method's own module, with its two ray-generation programs, each in
    // a pipeline of its own beside the renderer's miss and hit programs.
    OptixModule module = nullptr;
    OptixProgramGroup cameraGroup = nullptr;
    OptixProgramGroup photonGroup = nullptr;
    OptixPipeline cameraPipeline = nullptr;
    OptixPipeline photonPipeline = nullptr;
    // Their SBT records carry the method's parameters (LepskiParams), which
    // is how the programs get them without a change to Params.
    SbtRecord<LepskiParams> cameraRecord = {};
    SbtRecord<LepskiParams> photonRecord = {};
    CUdeviceptr dCameraRecord = 0;
    CUdeviceptr dPhotonRecord = 0;

    LepskiOptions options; // --lepski-*
    LepskiBuffers buffers;
    LepskiParams last = {}; // the parameters of the last iteration rendered
    int iteration = 0; // iterations since the last restart
    int saturatedFrames = 0; // passes that deposited more photons than fit

    // Sums over the frames rendered, for the report: deposits past the first
    // hit, those in an active cell, those stored.
    double candidates = 0.0, active = 0.0, stored = 0.0;
    long long frames = 0;
    // Pixels at each scale when the run ended, and (last) without a photon.
    unsigned int histogram[LEPSKI_K_SCALES + 1] = {};
    bool haveHistogram = false;
};
static LepskiHost gLepski;

// Created the first time the method renders a frame, so that a run of any
// other method does not even load its PTX.
static void lepskiCreatePipelines(RendererState& state)
{
    if (gLepski.module)
        return;

    std::ifstream f("sppm_lepski_device.ptx", std::ios::binary);
    const std::string ptx(std::istreambuf_iterator<char>(f), {});
    if (ptx.empty()) {
        std::cerr << "sppm_lepski_device.ptx not found next to the renderer — rebuild\n";
        std::exit(1);
    }

    // The options of createModule: the two modules meet in one pipeline.
    OptixModuleCompileOptions mco = {};
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    char log[2048];
    size_t ls = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(state.context, &mco, &state.pipelineCompileOptions,
        ptx.c_str(), ptx.size(), log, &ls, &gLepski.module));

    OptixProgramGroupOptions pgo = {};
    auto raygen = [&](const char* name, OptixProgramGroup& group) {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module = gLepski.module;
        d.raygen.entryFunctionName = name;
        ls = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context, &d, 1, &pgo, log, &ls, &group));
    };
    raygen("__raygen__lepski_camera", gLepski.cameraGroup);
    raygen("__raygen__lepski_photon", gLepski.photonGroup);

    // The renderer's own miss and hit groups, so its SBT serves these
    // pipelines too.
    const OptixProgramGroup camera[] = { gLepski.cameraGroup, state.missGroup,
        state.shadowMissGroup, state.hitGroup };
    gLepski.cameraPipeline = linkPipeline(state, camera, (uint32_t)std::size(camera));
    const OptixProgramGroup photon[] = { gLepski.photonGroup, state.missGroup,
        state.shadowMissGroup, state.hitGroup };
    gLepski.photonPipeline = linkPipeline(state, photon, (uint32_t)std::size(photon));

    OPTIX_CHECK(optixSbtRecordPackHeader(gLepski.cameraGroup, &gLepski.cameraRecord));
    OPTIX_CHECK(optixSbtRecordPackHeader(gLepski.photonGroup, &gLepski.photonRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gLepski.dCameraRecord),
        sizeof(gLepski.cameraRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gLepski.dPhotonRecord),
        sizeof(gLepski.photonRecord)));
}

// Frees the method's device memory and OptiX objects (the end of a run).
static void lepskiDestroy()
{
    lepskiFree(gLepski.buffers);
    if (gLepski.dCameraRecord)
        cudaFree(reinterpret_cast<void*>(gLepski.dCameraRecord));
    if (gLepski.dPhotonRecord)
        cudaFree(reinterpret_cast<void*>(gLepski.dPhotonRecord));
    if (gLepski.cameraPipeline)
        optixPipelineDestroy(gLepski.cameraPipeline);
    if (gLepski.photonPipeline)
        optixPipelineDestroy(gLepski.photonPipeline);
    if (gLepski.cameraGroup)
        optixProgramGroupDestroy(gLepski.cameraGroup);
    if (gLepski.photonGroup)
        optixProgramGroupDestroy(gLepski.photonGroup);
    if (gLepski.module)
        optixModuleDestroy(gLepski.module);
    const LepskiOptions options = gLepski.options;
    gLepski = LepskiHost();
    gLepski.options = options;
}

static unsigned int lepskiPhotonCapacity(const Params& p)
{
    if (gLepski.options.maxPhotons > 0)
        return (unsigned int)gLepski.options.maxPhotons;
    const size_t worst = size_t(p.num_photon_paths) * size_t(std::max(p.max_depth - 1, 1));
    return (unsigned int)std::min(worst, kLepskiMaxPhotonSlots);
}

// The mean luminance of the flux an emitted photon path carries: everything
// the emitters radiate, shared among the paths (__raygen__lepski_photon
// gives a path its emitter's flux over the probability of having picked it,
// which averages to exactly this).  What the storage roulette's threshold is
// a fraction of.
static float lepskiMeanPhotonLuminance(const Params& p, const std::vector<EmissiveTriangle>& lights)
{
    double flux = 0.0;
    for (const auto& lt : lights)
        flux += luminance(lt.emission) * lt.area * RT_PI; // Φ = L·A·π
    if (p.emit_sky_photons && p.sky_intensity > 0.f && p.scene_radius > 0.f) {
        // Φ_env = πR² ∫ L(ω) dω, as in computeSkySelectProb
        const int NT = 64, NP = 128;
        double sum = 0.0;
        for (int i = 0; i < NT; ++i) {
            const double ct = 1.0 - 2.0 * (i + 0.5) / NT;
            const double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
            for (int j = 0; j < NP; ++j) {
                const double phi = 2.0 * M_PI * (j + 0.5) / NP;
                sum += luminance(skyRadiance(make_float3((float)(st * std::cos(phi)), (float)ct,
                                                 (float)(st * std::sin(phi))),
                    p.sky_intensity));
            }
        }
        flux += sum / double(NT * NP) * 4.0 * M_PI * RT_PI * p.scene_radius * p.scene_radius;
    }
    return float(flux * p.photon_power_scale / double(std::max(p.num_photon_paths, 1)));
}

// The method's parameters for the frame about to be rendered.
static LepskiParams lepskiFrameParams(const RendererState& state, const Params& p)
{
    const LepskiOptions& o = gLepski.options;
    const float dx = p.scene_aabb_max.x - p.scene_aabb_min.x;
    const float dy = p.scene_aabb_max.y - p.scene_aabb_min.y;
    const float dz = p.scene_aabb_max.z - p.scene_aabb_min.z;
    const float diag = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1e-6f);

    LepskiParams L = {};
    L.width = p.width;
    L.height = p.height;
    L.accum_buffer = p.accum_buffer;
    L.frame_buffer = p.frame_buffer;
    L.frame_radiance = p.query_radiance;
    L.iteration = gLepski.iteration;
    L.view = o.view;
    L.kappa = o.kappa;
    if (p.stats) {
        L.stat_tested = p.stat_tested;
        L.stat_accepted = p.stat_accepted;
        L.stat_cells = p.stat_cells;
    }
    L.origin = p.scene_aabb_min;
    L.d0 = o.d0 * diag;
    L.h0 = o.h0 * L.d0;
    L.max_lod = o.maxLod;
    // --k-pix is the radius of the largest scale in pixel footprints, as it
    // is the other methods' radius: the default, 4, makes scale 1 one pixel.
    L.base_scale = p.k_pix / lepskiScale(LEPSKI_K_SCALES - 1);
    L.culling = o.culling ? 1 : 0;
    L.flux_threshold = o.fluxRR * lepskiMeanPhotonLuminance(p, state.hostLights);
    lepskiBind(gLepski.buffers, L);
    return L;
}

static void lepskiUploadParams(const LepskiParams& L, CUstream stream)
{
    gLepski.cameraRecord.data = L;
    gLepski.photonRecord.data = L;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(gLepski.dCameraRecord),
        &gLepski.cameraRecord, sizeof(gLepski.cameraRecord), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(gLepski.dPhotonRecord),
        &gLepski.photonRecord, sizeof(gLepski.photonRecord), cudaMemcpyHostToDevice, stream));
}

// Stage 2: the photon trace, and the read-back of what it counted.
static void lepskiTracePhotons(RendererState& state, const Params& p, const LepskiParams& L,
    CUstream stream, unsigned int counters[LEPSKI_COUNTERS])
{
    OptixShaderBindingTable sbt = state.sbt;
    sbt.raygenRecord = gLepski.dPhotonRecord;
    OPTIX_CHECK(optixLaunch(gLepski.photonPipeline, stream, state.dParams, sizeof(Params),
        &sbt, (unsigned)p.num_photon_paths, 1, 1));
    CUDA_CHECK(cudaMemcpyAsync(counters, L.counters, LEPSKI_COUNTERS * sizeof(unsigned int),
        cudaMemcpyDeviceToHost, stream));
}

// One iteration.  `restart` begins a new running estimate (the first frame
// of a run, or the frame after the camera or a setting changed).
//
// The four stages are timed like the other methods' passes and reported in
// the same fields: camera, photon_trace, build and query.
//
// `verify` (--verify-gather): the iteration's gather is also run by brute
// force over a photon map traced without culling — every photon stored,
// every photon tested for every pixel — and compared with the gather through
// the grid over the culled map.  Culling may only drop photons no gather can
// reach, and the grid may only change which photons are looked at, so the
// two must agree to rounding.  Such a run checks; its times mean nothing.
static void launchLepskiFrame(RendererState& state, Params& p, CUstream& stream,
    FrameTimings* ft, bool restart, GatherVerification* verify = nullptr)
{
    static GpuTimer cameraTimer, photonTimer, buildTimer, gatherTimer;
    static std::vector<float3> verifyBrute, verifyGrid;

    lepskiCreatePipelines(state);
    // No allocation in a steady-state frame: only when the image or the
    // photon capacity changed, which also restarts the estimate.
    if (lepskiAllocate(gLepski.buffers, p.width, p.height, lepskiPhotonCapacity(p)))
        restart = true;
    if (restart)
        gLepski.iteration = 0;
    ++gLepski.iteration;

    p.samples_per_pixel = 1; // one camera sample per iteration, as in every photon mapper here

    const LepskiParams L = lepskiFrameParams(state, p);
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams), &p, sizeof(Params),
        cudaMemcpyHostToDevice, stream));
    lepskiUploadParams(L, stream);

    // Stage 1 — the camera walk; it marks the cells its gathers will search.
    OptixShaderBindingTable sbt = state.sbt;
    sbt.raygenRecord = gLepski.dCameraRecord;
    cameraTimer.start(stream);
    lepskiBeginIteration(L, stream);
    OPTIX_CHECK(optixLaunch(gLepski.cameraPipeline, stream, state.dParams, sizeof(Params),
        &sbt, p.width, p.height, 1));
    const float cameraMs = cameraTimer.stopMs(stream);

    unsigned int counters[LEPSKI_COUNTERS] = {};
    if (verify) {
        // The reference: no culling, brute force.
        LepskiParams all = L;
        all.culling = 0;
        lepskiUploadParams(all, stream);
        lepskiTracePhotons(state, p, all, stream, counters);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        lepskiBuildGrid(all, gLepski.buffers, std::min(counters[LEPSKI_COUNT_STORED], L.max_stored), stream);
        const size_t pixels = size_t(p.width) * p.height;
        verifyBrute.resize(pixels);
        verifyGrid.resize(pixels);
        lepskiProbe(all, true, p.query_radiance, stream);
        CUDA_CHECK(cudaMemcpy(verifyBrute.data(), p.query_radiance, pixels * sizeof(float3),
            cudaMemcpyDeviceToHost));
        lepskiUploadParams(L, stream);
        lepskiClearCounters(L, stream);
    }

    // Stage 2 — the photon trace.  The number of photons it stored is what
    // stage 3 sorts, so reading it back is part of the pass.
    photonTimer.start(stream);
    lepskiTracePhotons(state, p, L, stream, counters);
    const float photonMs = photonTimer.stopMs(stream);

    const unsigned int stored = std::min(counters[LEPSKI_COUNT_STORED], L.max_stored);
    if (counters[LEPSKI_COUNT_STORED] > L.max_stored) {
        // The photons that did not fit are missing from this iteration's
        // estimate, so say so rather than render a darker image quietly.
        if (gLepski.saturatedFrames++ == 0)
            std::cerr << "\n[sppm_lepski] photon buffer saturated: "
                      << counters[LEPSKI_COUNT_STORED] << " photons, room for "
                      << L.max_stored << " — fewer --paths, a smaller --depth,"
                                         " or --lepski-max-photons\n";
    }

    // Stage 3 — the counting sort into the hash table.
    buildTimer.start(stream);
    lepskiBuildGrid(L, gLepski.buffers, stored, stream);
    const float buildMs = buildTimer.stopMs(stream);

    if (verify) {
        lepskiProbe(L, false, p.query_radiance, stream);
        CUDA_CHECK(cudaMemcpy(verifyGrid.data(), p.query_radiance,
            verifyGrid.size() * sizeof(float3), cudaMemcpyDeviceToHost));
        *verify = compareRadiance(verifyBrute, verifyGrid);
    }

    // Stage 4 — the gather, the running estimate and the image.
    gatherTimer.start(stream);
    lepskiGather(L, stream);
    const float gatherMs = gatherTimer.stopMs(stream);

    gLepski.last = L;
    gLepski.candidates += counters[LEPSKI_COUNT_CANDIDATES];
    gLepski.active += counters[LEPSKI_COUNT_ACTIVE];
    gLepski.stored += stored;
    ++gLepski.frames;

    if (ft) {
        // The radii, for the record only, after everything has been timed —
        // as the other methods' footprint radii are reduced.
        LepskiRadiusStats radii;
        lepskiRadiusStats(L, gLepski.buffers, p.pixel_angle, stream, radii);
        ft->r_max = radii.max_r;
        ft->r_mean = radii.valid ? float(radii.sum_r / double(radii.valid)) : 0.f;
        ft->r_px_mean = radii.valid ? float(radii.sum_px / double(radii.valid)) : 0.f;
        ft->radius = radii.max_r;

        ft->camera = cameraMs;
        ft->photon_trace = photonMs;
        ft->build = buildMs;
        ft->query = gatherMs;
        ft->photons = (int)stored;
        ft->deposits = (long long)counters[LEPSKI_COUNT_CANDIDATES];
        ft->culled = counters[LEPSKI_COUNT_CANDIDATES] > 0
            ? float(1.0 - double(counters[LEPSKI_COUNT_ACTIVE]) / double(counters[LEPSKI_COUNT_CANDIDATES]))
            : 0.f;
        ft->buckets = (long long)LEPSKI_HASH_SIZE;
        // what the gather reads: the offsets and the photons in bucket order
        ft->bytes = (long long)(((size_t)LEPSKI_HASH_SIZE + 1) * sizeof(unsigned int)
            + (size_t)stored * sizeof(PackedPhoton));
    }
}

static bool lepskiCullingOn() { return gLepski.options.culling; }

// The method's own lines under the timing summary of a run, and the scale
// histogram the timing record gets: which scale every pixel's estimate
// stands at when the run ends.  One more kernel and a handful of scalars
// read back, after everything has been timed.
static void lepskiReport(const Params& p)
{
    if (gLepski.frames == 0)
        return;
    lepskiScaleHistogram(gLepski.last, gLepski.buffers, gLepski.histogram, 0);
    gLepski.haveHistogram = true;

    const LepskiOptions& o = gLepski.options;
    const double inv = 1.0 / double(gLepski.frames);
    const double candidates = std::max(gLepski.candidates, 1.0);
    std::cout << "         sppm_lepski: per frame " << gLepski.candidates * inv
              << " deposits past the first hit, " << 100.0 * gLepski.active / candidates
              << "% in an active cell" << (o.culling ? "" : " (culling off)")
              << ", " << gLepski.stored * inv << " stored after the flux roulette"
              << " (alive / candidates = " << gLepski.stored / candidates << ")\n"
              << "         grid: d0 " << gLepski.last.d0 << ", level-0 cell " << gLepski.last.h0
              << " (" << o.h0 << " d0), " << o.maxLod << " levels, " << LEPSKI_HASH_SIZE
              << " buckets; roulette below " << o.fluxRR << " of the mean photon flux;"
              << " kappa " << o.kappa << "; view " << lepskiViewName(o.view) << "\n";
    const double pixels = std::max(1.0, double(p.width) * double(p.height));
    std::cout << "         scale when the run ended (0.25, 0.5, 1, 2, 4 x the base radius):";
    for (int k = 0; k < LEPSKI_K_SCALES; ++k)
        std::cout << " " << 100.0 * gLepski.histogram[k] / pixels << "%";
    std::cout << "; no photon yet: " << 100.0 * gLepski.histogram[LEPSKI_K_SCALES] / pixels << "%\n";
    LepskiKernelInfo kernels[4];
    const int count = lepskiKernelInfo(kernels);
    for (int i = 0; i < count; ++i)
        if (std::string(kernels[i].name) == "lepski_gather")
            std::cout << "         gather kernel: " << kernels[i].registers << " registers, "
                      << kernels[i].local_bytes << " bytes of local memory per thread\n";
    std::cout << "         device memory of the method: "
              << double(gLepski.buffers.bytes) / (1024.0 * 1024.0) << " MiB, "
              << gLepski.buffers.max_stored << " photon slots\n";
    if (gLepski.saturatedFrames > 0)
        std::cout << "         SATURATED: " << gLepski.saturatedFrames
                  << " frames deposited more photons than fit; their estimate is short of them\n";
}

// The method's part of the timing record: its settings, and what a run did.
static void lepskiWriteJson(std::ostream& out)
{
    const LepskiOptions& o = gLepski.options;
    const double inv = gLepski.frames > 0 ? 1.0 / double(gLepski.frames) : 0.0;
    out << "  \"lepski\": {\"d0\": " << gLepski.last.d0 << ", \"h0\": " << gLepski.last.h0
        << ", \"d0_fraction\": " << o.d0 << ", \"h0_fraction\": " << o.h0
        << ", \"max_lod\": " << o.maxLod << ", \"hash_size\": " << LEPSKI_HASH_SIZE
        << ", \"scales\": " << LEPSKI_K_SCALES << ", \"kappa\": " << o.kappa
        << ", \"flux_rr\": " << o.fluxRR << ", \"culling\": " << (o.culling ? "true" : "false")
        << ", \"view\": \"" << lepskiViewName(o.view) << "\""
        << ", \"photon_slots\": " << gLepski.buffers.max_stored
        << ", \"device_bytes\": " << gLepski.buffers.bytes
        << ", \"candidates_mean\": " << gLepski.candidates * inv
        << ", \"active_mean\": " << gLepski.active * inv
        << ", \"stored_mean\": " << gLepski.stored * inv
        << ", \"alive_over_candidates\": "
        << (gLepski.candidates > 0.0 ? gLepski.stored / gLepski.candidates : 0.0)
        << ", \"saturated_frames\": " << gLepski.saturatedFrames
        << ", \"scale_histogram\": [";
    for (int k = 0; k <= LEPSKI_K_SCALES; ++k)
        out << (k ? ", " : "") << (gLepski.haveHistogram ? gLepski.histogram[k] : 0u);
    out << "], \"kernels\": {";
    LepskiKernelInfo kernels[4];
    const int count = lepskiKernelInfo(kernels);
    for (int i = 0; i < count; ++i)
        out << (i ? ", " : "") << "\"" << kernels[i].name << "\": {\"registers\": "
            << kernels[i].registers << ", \"local_bytes\": " << kernels[i].local_bytes << "}";
    out << "}},\n";
}

// The method's controls in the realtime GUI.  None of them restarts the
// running estimate: the view and kappa only change which of the scales'
// means is shown, and culling changes which photons are stored, never which
// ones a gather finds.
static void lepskiGui(const FrameTimings& last)
{
    LepskiOptions& o = gLepski.options;
    const char* views[] = { "Normal: direct + indirect (selected scale)",
        "Indirect only (selected scale)", "Scale heatmap (blue 0.25x .. red 4x)",
        "Fixed base scale: direct + indirect (1x)" };
    ImGui::Combo("View", &o.view, views, IM_ARRAYSIZE(views));
    ImGui::SliderFloat("kappa", &o.kappa, 0.5f, 6.f, "%.2f standard errors");
    ImGui::Checkbox("Active-cell culling", &o.culling);
    ImGui::Text("Stored %d of %lld deposits (%.1f%% outside active cells), iteration %d",
        last.photons, last.deposits, 100.0 * last.culled, gLepski.iteration);
}
