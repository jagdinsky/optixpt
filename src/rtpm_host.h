#pragma once
// The host side of RTPM (rtpm.h): its OptiX module, pipelines and shader
// binding tables, the photon acceleration structures it rebuilds every
// iteration, and the iteration's passes.
//
// Part of main.cpp's translation unit — included there once, after the
// renderer state, the pipeline helpers, the timers and the image comparison
// it uses — so that the method's host code is all in this file and main.cpp
// only registers it.
#include "rtpm.h"

// A path deposits at most one photon per bounce after its first, and at most
// one of those is a caustic photon; that bounds the maps.  Beyond this many
// photons per map the bound is not allocated in full, and a pass that
// outgrows the buffers is reported instead.
static constexpr size_t kRtpmMaxPhotonSlots = 8'000'000;

struct RtpmHost {
    // The method's own module.  It is compiled for pipelines that hold custom
    // primitives besides triangles, which the renderer's module is not, so
    // its pipelines use this module's copies of the miss and hit programs.
    OptixPipelineCompileOptions compileOptions = {};
    OptixModule module = nullptr;
    OptixProgramGroup photonRaygen = nullptr;
    OptixProgramGroup collectRaygen = nullptr;
    OptixProgramGroup sceneMiss = nullptr; // primary rays of the photon pass
    OptixProgramGroup shadowMiss = nullptr;
    OptixProgramGroup sceneHit = nullptr;
    OptixProgramGroup collectMiss = nullptr; // empty: collection rays hit nothing
    OptixProgramGroup photonHit = nullptr; // intersection + any-hit of a photon box
    // One pipeline per pass, as everywhere in the renderer.
    OptixPipeline photonPipeline = nullptr;
    OptixPipeline collectPipeline = nullptr;

    // The ray-generation records carry the method's parameters (RtpmParams),
    // the photon hit records the map they belong to (RtpmMapRecord).
    SbtRecord<RtpmParams> photonRecord = {};
    SbtRecord<RtpmParams> collectRecord = {};
    SbtRecord<RtpmMapRecord> mapRecords[RTPM_MAPS] = {};
    CUdeviceptr dPhotonRecord = 0;
    CUdeviceptr dCollectRecord = 0;
    CUdeviceptr dSceneMiss = 0;
    CUdeviceptr dSceneHit = 0;
    CUdeviceptr dCollectMiss = 0;
    CUdeviceptr dPhotonHit = 0;
    OptixShaderBindingTable sbtPhoton = {};
    OptixShaderBindingTable sbtCollect = {};

    // The photon structures' memory, sized once for the maps' capacities: an
    // iteration builds into it and allocates nothing.  The two builds run
    // one after the other on the stream and share the working memory.
    CUdeviceptr dAabbs[RTPM_MAPS] = {}; // the build inputs' box buffers
    CUdeviceptr dAccelOut[RTPM_MAPS] = {};
    size_t accelOutBytes[RTPM_MAPS] = {};
    CUdeviceptr dAccelTemp = 0;
    size_t accelTempBytes = 0;
    unsigned int accelCapacity[RTPM_MAPS] = {};

    RtpmOptions options; // --rtpm-*
    RtpmBuffers buffers;
    int saturatedFrames = 0; // passes that deposited more photons than fit

    // Sums over the frames rendered, for the report.
    double candidates = 0.0, visible = 0.0, stored[RTPM_MAPS] = {};
    double accelBytes = 0.0;
    long long frames = 0;
};
static RtpmHost gRtpm;

static const unsigned int kRtpmGeometryFlags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;

// The build input of a map over its first `count` boxes.  Every photon is
// reported to the any-hit program exactly once (paper sec. 3.1): the sum must
// not hold a photon twice.
static OptixBuildInput rtpmBuildInput(int m, unsigned int count)
{
    OptixBuildInput in = {};
    in.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    in.customPrimitiveArray.aabbBuffers = &gRtpm.dAabbs[m];
    in.customPrimitiveArray.numPrimitives = count;
    in.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    in.customPrimitiveArray.flags = &kRtpmGeometryFlags;
    in.customPrimitiveArray.numSbtRecords = 1;
    return in;
}

// Rebuilt from scratch every iteration, so the build that is fastest to make
// (paper sec. 3.5: photons at the same index move too far for a refit).
static OptixAccelBuildOptions rtpmBuildOptions()
{
    OptixAccelBuildOptions options = {};
    options.buildFlags = gRtpm.options.build == 2 ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                                                  : (gRtpm.options.build == 1 ? OPTIX_BUILD_FLAG_NONE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD);
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    return options;
}

// Created the first time the method renders a frame, so that a run of any
// other method does not even load its PTX.
static void rtpmCreatePipelines(RendererState& state)
{
    if (gRtpm.module)
        return;

    std::ifstream f("rtpm_device.ptx", std::ios::binary);
    const std::string ptx(std::istreambuf_iterator<char>(f), {});
    if (ptx.empty()) {
        std::cerr << "rtpm_device.ptx not found next to the renderer — rebuild\n";
        std::exit(1);
    }

    // The renderer's options, plus the custom primitives the photons are.
    gRtpm.compileOptions = state.pipelineCompileOptions;
    gRtpm.compileOptions.usesPrimitiveTypeFlags = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE
        | (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OptixModuleCompileOptions mco = {};
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    char log[2048];
    size_t ls = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(state.context, &mco, &gRtpm.compileOptions,
        ptx.c_str(), ptx.size(), log, &ls, &gRtpm.module));

    OptixProgramGroupOptions pgo = {};
    auto create = [&](const OptixProgramGroupDesc& d, OptixProgramGroup& group) {
        ls = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context, &d, 1, &pgo, log, &ls, &group));
    };
    auto raygen = [&](const char* name, OptixProgramGroup& group) {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module = gRtpm.module;
        d.raygen.entryFunctionName = name;
        create(d, group);
    };
    auto miss = [&](const char* name, OptixProgramGroup& group) {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        d.miss.module = gRtpm.module;
        d.miss.entryFunctionName = name;
        create(d, group);
    };
    raygen("__raygen__rtpm_photon", gRtpm.photonRaygen);
    raygen("__raygen__rtpm_collect", gRtpm.collectRaygen);
    miss("__miss__ms", gRtpm.sceneMiss);
    miss("__miss__shadow", gRtpm.shadowMiss);
    miss("__miss__rtpm", gRtpm.collectMiss);
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        d.hitgroup.moduleCH = gRtpm.module;
        d.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        create(d, gRtpm.sceneHit);
    }
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        d.hitgroup.moduleIS = gRtpm.module;
        d.hitgroup.entryFunctionNameIS = "__intersection__rtpm";
        d.hitgroup.moduleAH = gRtpm.module;
        d.hitgroup.entryFunctionNameAH = "__anyhit__rtpm";
        create(d, gRtpm.photonHit);
    }

    // linkPipeline links with the renderer's compile options; these pipelines
    // have their own, so they stand in for the two calls.
    const OptixPipelineCompileOptions renderers = state.pipelineCompileOptions;
    state.pipelineCompileOptions = gRtpm.compileOptions;
    const OptixProgramGroup photon[] = { gRtpm.photonRaygen, gRtpm.sceneMiss, gRtpm.shadowMiss,
        gRtpm.sceneHit };
    gRtpm.photonPipeline = linkPipeline(state, photon, (uint32_t)std::size(photon));
    const OptixProgramGroup collect[] = { gRtpm.collectRaygen, gRtpm.collectMiss, gRtpm.photonHit };
    gRtpm.collectPipeline = linkPipeline(state, collect, (uint32_t)std::size(collect));
    state.pipelineCompileOptions = renderers;

    // The shader binding tables.  The photon pass traces the scene: the
    // renderer's layout — primary and shadow miss, one hit record per
    // material — with this module's programs.  The collection pass traces
    // the photon structures: one empty miss, one hit record per map.
    OPTIX_CHECK(optixSbtRecordPackHeader(gRtpm.photonRaygen, &gRtpm.photonRecord));
    OPTIX_CHECK(optixSbtRecordPackHeader(gRtpm.collectRaygen, &gRtpm.collectRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gRtpm.dPhotonRecord), sizeof(gRtpm.photonRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gRtpm.dCollectRecord), sizeof(gRtpm.collectRecord)));

    const size_t missStride = roundUp(sizeof(SbtRecord<MissData>), OPTIX_SBT_RECORD_ALIGNMENT);
    SbtRecord<MissData> sceneMiss[2] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(gRtpm.sceneMiss, &sceneMiss[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(gRtpm.shadowMiss, &sceneMiss[1]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gRtpm.dSceneMiss), missStride * 2));
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(gRtpm.dSceneMiss + i * missStride),
            &sceneMiss[i], sizeof(sceneMiss[i]), cudaMemcpyHostToDevice));
    SbtRecord<MissData> collectMiss = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(gRtpm.collectMiss, &collectMiss));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gRtpm.dCollectMiss), missStride));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(gRtpm.dCollectMiss), &collectMiss,
        sizeof(collectMiss), cudaMemcpyHostToDevice));

    const uint32_t materials = std::max(1u, state.numMaterials);
    const size_t hitStride = roundUp(sizeof(SbtRecord<HitData>), OPTIX_SBT_RECORD_ALIGNMENT);
    std::vector<SbtRecord<HitData>> sceneHit(materials);
    for (uint32_t i = 0; i < materials; ++i) {
        OPTIX_CHECK(optixSbtRecordPackHeader(gRtpm.sceneHit, &sceneHit[i]));
        sceneHit[i].data.albedo = (i < state.hostMaterials.size()) ? state.hostMaterials[i].albedo
                                                                   : make_float3(0.5f, 0.5f, 0.5f);
        sceneHit[i].data.emission = (i < state.hostMaterials.size()) ? state.hostMaterials[i].emission
                                                                     : make_float3(0.f, 0.f, 0.f);
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gRtpm.dSceneHit), hitStride * materials));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(gRtpm.dSceneHit), sceneHit.data(),
        hitStride * materials, cudaMemcpyHostToDevice));

    for (int m = 0; m < RTPM_MAPS; ++m)
        OPTIX_CHECK(optixSbtRecordPackHeader(gRtpm.photonHit, &gRtpm.mapRecords[m]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gRtpm.dPhotonHit), sizeof(gRtpm.mapRecords)));

    gRtpm.sbtPhoton.raygenRecord = gRtpm.dPhotonRecord;
    gRtpm.sbtPhoton.missRecordBase = gRtpm.dSceneMiss;
    gRtpm.sbtPhoton.missRecordStrideInBytes = (unsigned int)missStride;
    gRtpm.sbtPhoton.missRecordCount = 2;
    gRtpm.sbtPhoton.hitgroupRecordBase = gRtpm.dSceneHit;
    gRtpm.sbtPhoton.hitgroupRecordStrideInBytes = (unsigned int)hitStride;
    gRtpm.sbtPhoton.hitgroupRecordCount = materials;

    gRtpm.sbtCollect.raygenRecord = gRtpm.dCollectRecord;
    gRtpm.sbtCollect.missRecordBase = gRtpm.dCollectMiss;
    gRtpm.sbtCollect.missRecordStrideInBytes = (unsigned int)missStride;
    gRtpm.sbtCollect.missRecordCount = 1;
    gRtpm.sbtCollect.hitgroupRecordBase = gRtpm.dPhotonHit;
    gRtpm.sbtCollect.hitgroupRecordStrideInBytes = (unsigned int)sizeof(SbtRecord<RtpmMapRecord>);
    gRtpm.sbtCollect.hitgroupRecordCount = RTPM_MAPS;
}

static void rtpmFreeAccelMemory()
{
    for (int m = 0; m < RTPM_MAPS; ++m) {
        if (gRtpm.dAccelOut[m])
            cudaFree(reinterpret_cast<void*>(gRtpm.dAccelOut[m]));
        gRtpm.dAccelOut[m] = 0;
        gRtpm.accelOutBytes[m] = 0;
        gRtpm.accelCapacity[m] = 0;
    }
    if (gRtpm.dAccelTemp)
        cudaFree(reinterpret_cast<void*>(gRtpm.dAccelTemp));
    gRtpm.dAccelTemp = 0;
    gRtpm.accelTempBytes = 0;
}

// Frees the method's device memory (another structure was chosen in the GUI).
static void rtpmRelease()
{
    rtpmFree(gRtpm.buffers);
    rtpmFreeAccelMemory();
}

// Frees the method's device memory and OptiX objects (the end of a run).
static void rtpmDestroy()
{
    rtpmRelease();
    for (CUdeviceptr* p : { &gRtpm.dPhotonRecord, &gRtpm.dCollectRecord, &gRtpm.dSceneMiss,
             &gRtpm.dSceneHit, &gRtpm.dCollectMiss, &gRtpm.dPhotonHit })
        if (*p)
            cudaFree(reinterpret_cast<void*>(*p));
    if (gRtpm.photonPipeline)
        optixPipelineDestroy(gRtpm.photonPipeline);
    if (gRtpm.collectPipeline)
        optixPipelineDestroy(gRtpm.collectPipeline);
    for (OptixProgramGroup g : { gRtpm.photonRaygen, gRtpm.collectRaygen, gRtpm.sceneMiss,
             gRtpm.shadowMiss, gRtpm.sceneHit, gRtpm.collectMiss, gRtpm.photonHit })
        if (g)
            optixProgramGroupDestroy(g);
    if (gRtpm.module)
        optixModuleDestroy(gRtpm.module);
    const RtpmOptions options = gRtpm.options;
    gRtpm = RtpmHost();
    gRtpm.options = options;
}

// Makes the photon buffers and the structures' memory fit the run: no
// allocation in a steady-state frame, only when the photon capacity changed.
static void rtpmPrepare(RendererState& state, const Params& p)
{
    const size_t limit = gRtpm.options.maxPhotons > 0 ? size_t(gRtpm.options.maxPhotons)
                                                      : kRtpmMaxPhotonSlots;
    const size_t paths = size_t(p.num_photon_paths);
    const unsigned int caustic = (unsigned int)std::min(paths, limit);
    const unsigned int global = (unsigned int)std::min(paths * size_t(std::max(p.max_depth - 1, 1)), limit);
    rtpmAllocate(gRtpm.buffers, caustic, global, (unsigned int)gRtpm.options.cullBits);

    const OptixAccelBuildOptions options = rtpmBuildOptions();
    bool resized = false;
    for (int m = 0; m < RTPM_MAPS; ++m)
        resized |= gRtpm.accelCapacity[m] != gRtpm.buffers.capacity[m];
    if (!resized)
        return;

    rtpmFreeAccelMemory();
    size_t temp = 0;
    for (int m = 0; m < RTPM_MAPS; ++m) {
        gRtpm.dAabbs[m] = reinterpret_cast<CUdeviceptr>(gRtpm.buffers.aabbs[m]);
        const OptixBuildInput in = rtpmBuildInput(m, gRtpm.buffers.capacity[m]);
        OptixAccelBufferSizes sizes = {};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &options, &in, 1, &sizes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gRtpm.dAccelOut[m]), sizes.outputSizeInBytes));
        gRtpm.accelOutBytes[m] = sizes.outputSizeInBytes;
        gRtpm.accelCapacity[m] = gRtpm.buffers.capacity[m];
        temp = std::max(temp, (size_t)sizes.tempSizeInBytes);
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gRtpm.dAccelTemp), temp));
    gRtpm.accelTempBytes = temp;
}

// The method's parameters for the frame whose camera pass has just run:
// p.max_gather_radius is the largest radius any of its gathers uses — the one
// radius of the frame, or with --radius-mode footprint the largest pixel's —
// and the boxes are that wide.
static RtpmParams rtpmFrameParams(const Params& p)
{
    const RtpmOptions& o = gRtpm.options;
    const float rMax = p.max_gather_radius > 0.f ? p.max_gather_radius : p.gather_radius;

    RtpmParams R = {};
    rtpmBind(gRtpm.buffers, R);
    R.maps[RTPM_GLOBAL].radius_scale = 1.f;
    R.maps[RTPM_GLOBAL].box_radius = rMax;
    R.maps[RTPM_CAUSTIC].radius_scale = o.causticScale;
    R.maps[RTPM_CAUSTIC].box_radius = rMax * o.causticScale;
    R.culling = o.culling ? 1 : 0;
    // A cell as wide as a global photon's box (paper sec. 3.7), and a part
    // in a thousand wider, so that the rounding of a cell coordinate cannot
    // leave a photon at the very edge of a gather's radius in a ninth cell.
    R.cull_cell = 2.f * rMax * 1.001f;
    R.origin = p.scene_aabb_min;
    R.reject = o.reject;
    R.list_size = o.listSize;
    R.uniform_kernel = o.uniformKernel ? 1 : 0;
    R.stats = p.stats;
    return R;
}

// The photon generation pass, and the read-back of what it counted.
static void rtpmTracePhotons(RendererState& state, const Params& p, const RtpmParams& R,
    CUstream stream, unsigned int counters[RTPM_COUNTERS])
{
    gRtpm.photonRecord.data = R;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(gRtpm.dPhotonRecord), &gRtpm.photonRecord,
        sizeof(gRtpm.photonRecord), cudaMemcpyHostToDevice, stream));
    rtpmClearCounters(R, stream);
    OPTIX_CHECK(optixLaunch(gRtpm.photonPipeline, stream, state.dParams, sizeof(Params),
        &gRtpm.sbtPhoton, (unsigned)p.num_photon_paths, 1, 1));
    CUDA_CHECK(cudaMemcpyAsync(counters, R.counters, RTPM_COUNTERS * sizeof(unsigned int),
        cudaMemcpyDeviceToHost, stream));
}

// The maps' photon counts once the pass has been read back; true if a map
// was offered more photons than it has room for.
static bool rtpmSetStored(RtpmParams& R, const unsigned int counters[RTPM_COUNTERS])
{
    bool saturated = false;
    for (int m = 0; m < RTPM_MAPS; ++m) {
        R.maps[m].stored = std::min(counters[m], R.maps[m].capacity);
        saturated |= counters[m] > R.maps[m].capacity;
    }
    return saturated;
}

// The acceleration structures over this iteration's photons, into the memory
// set aside for them; returns the bytes the two structures need.
static size_t rtpmBuildStructures(RendererState& state, RtpmParams& R, CUstream stream)
{
    const OptixAccelBuildOptions options = rtpmBuildOptions();
    size_t bytes = 0;
    for (int m = 0; m < RTPM_MAPS; ++m) {
        RtpmMap& map = R.maps[m];
        map.handle = 0;
        if (map.stored == 0u)
            continue;
        const OptixBuildInput in = rtpmBuildInput(m, map.stored);
        OptixAccelBufferSizes sizes = {};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &options, &in, 1, &sizes));
        if (sizes.outputSizeInBytes > gRtpm.accelOutBytes[m]
            || sizes.tempSizeInBytes > gRtpm.accelTempBytes) {
            std::cerr << "[rtpm] a structure over " << map.stored << " photons needs more memory"
                                                                     " than was set aside for "
                      << map.capacity << "\n";
            std::exit(1);
        }
        bytes += sizes.outputSizeInBytes;
        OPTIX_CHECK(optixAccelBuild(state.context, stream, &options, &in, 1,
            gRtpm.dAccelTemp, gRtpm.accelTempBytes, gRtpm.dAccelOut[m], gRtpm.accelOutBytes[m],
            &map.handle, nullptr, 0));
    }
    return bytes;
}

// The collection pass over the structures R names, into p.query_radiance.
static void rtpmCollect(RendererState& state, const Params& p, const RtpmParams& R, CUstream stream)
{
    gRtpm.collectRecord.data = R;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(gRtpm.dCollectRecord), &gRtpm.collectRecord,
        sizeof(gRtpm.collectRecord), cudaMemcpyHostToDevice, stream));
    for (int m = 0; m < RTPM_MAPS; ++m) {
        gRtpm.mapRecords[m].data.photons = R.maps[m].photons;
        gRtpm.mapRecords[m].data.list_size = R.list_size;
        gRtpm.mapRecords[m].data.uniform_kernel = R.uniform_kernel;
        gRtpm.mapRecords[m].data.stats = R.stats;
    }
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(gRtpm.dPhotonHit), gRtpm.mapRecords,
        sizeof(gRtpm.mapRecords), cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(gRtpm.collectPipeline, stream, state.dParams, sizeof(Params),
        &gRtpm.sbtCollect, p.width, p.height, 1));
}

// One iteration after the camera pass (the renderer's own, which leaves the
// gather points and the frame's largest radius): the culling mask, the
// photon generation pass, the two structures, the collection pass.  They are
// timed like the other methods' passes and reported in the same fields:
// mark, photon_trace, build and query.  The accumulation is the renderer's.
//
// `verify` (--verify-gather): the collection is also done by brute force over
// maps traced without culling — every photon stored, every photon tested for
// every pixel — and compared with the collection through the structures over
// the culled maps, everything evaluated.  Culling may only drop photons no
// gather can reach, and the structures may only change which photons are
// looked at, so the two must agree to rounding.  Such a run checks; its
// times mean nothing.
static void launchRtpmPasses(RendererState& state, Params& p, CUstream& stream,
    FrameTimings* ft, GatherVerification* verify = nullptr)
{
    static GpuTimer markTimer, photonTimer, buildTimer, collectTimer;
    static std::vector<float3> verifyBrute, verifyCollect;

    rtpmCreatePipelines(state);
    rtpmPrepare(state, p);

    RtpmParams R = rtpmFrameParams(p);
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams), &p, sizeof(Params),
        cudaMemcpyHostToDevice, stream));

    // Photon Culling — the cells around this iteration's gather points.
    float markMs = 0.f;
    if (R.culling) {
        markTimer.start(stream);
        rtpmMarkCells(p, R, stream);
        markMs = markTimer.stopMs(stream);
    }

    unsigned int counters[RTPM_COUNTERS] = {};
    const size_t pixels = size_t(p.width) * p.height;
    if (verify) {
        // The reference: no culling, no structures.
        RtpmParams all = R;
        all.culling = 0;
        rtpmTracePhotons(state, p, all, stream, counters);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        rtpmSetStored(all, counters);
        verifyBrute.resize(pixels);
        verifyCollect.resize(pixels);
        rtpmBruteCollect(p, all, p.query_radiance, stream);
        CUDA_CHECK(cudaMemcpy(verifyBrute.data(), p.query_radiance, pixels * sizeof(float3),
            cudaMemcpyDeviceToHost));
    }

    // Photon generation.  The number of photons it stored is what the
    // structures are built over, so reading it back is part of the pass.
    photonTimer.start(stream);
    rtpmTracePhotons(state, p, R, stream, counters);
    const float photonMs = photonTimer.stopMs(stream);
    if (rtpmSetStored(R, counters) && gRtpm.saturatedFrames++ == 0)
        std::cerr << "\n[rtpm] photon buffers saturated: " << counters[RTPM_CAUSTIC] << " caustic and "
                  << counters[RTPM_GLOBAL] << " global photons, room for "
                  << R.maps[RTPM_CAUSTIC].capacity << " and " << R.maps[RTPM_GLOBAL].capacity
                  << " — fewer --paths, a smaller --depth, or --rtpm-max-photons\n";

    // The acceleration structures.
    buildTimer.start(stream);
    const size_t accelBytes = rtpmBuildStructures(state, R, stream);
    const float buildMs = buildTimer.stopMs(stream);

    if (verify) {
        RtpmParams full = R;
        full.list_size = 0;
        rtpmCollect(state, p, full, stream);
        CUDA_CHECK(cudaMemcpy(verifyCollect.data(), p.query_radiance, pixels * sizeof(float3),
            cudaMemcpyDeviceToHost));
        *verify = compareRadiance(verifyBrute, verifyCollect);
    }

    // Photon collection.
    collectTimer.start(stream);
    rtpmCollect(state, p, R, stream);
    const float collectMs = collectTimer.stopMs(stream);

    const unsigned int stored = R.maps[RTPM_CAUSTIC].stored + R.maps[RTPM_GLOBAL].stored;
    gRtpm.candidates += counters[RTPM_COUNT_CANDIDATES];
    gRtpm.visible += counters[RTPM_COUNT_VISIBLE];
    for (int m = 0; m < RTPM_MAPS; ++m)
        gRtpm.stored[m] += R.maps[m].stored;
    gRtpm.accelBytes += double(accelBytes);
    ++gRtpm.frames;

    if (ft) {
        ft->mark = markMs;
        ft->photon_trace = photonMs;
        ft->build = buildMs;
        ft->query = collectMs;
        ft->radius = R.maps[RTPM_GLOBAL].box_radius;
        ft->photons = (int)stored;
        ft->deposits = (long long)counters[RTPM_COUNT_CANDIDATES];
        ft->culled = counters[RTPM_COUNT_CANDIDATES] > 0
            ? float(1.0 - double(counters[RTPM_COUNT_VISIBLE]) / double(counters[RTPM_COUNT_CANDIDATES]))
            : 0.f;
        // what the collection reads: the two structures and the photon records
        ft->bytes = (long long)(accelBytes + size_t(stored) * sizeof(PackedPhoton));
    }
}

static bool rtpmCullingOn() { return gRtpm.options.culling; }

// The method's own lines under the timing summary of a run.
static void rtpmReport()
{
    if (gRtpm.frames == 0)
        return;
    const RtpmOptions& o = gRtpm.options;
    const double inv = 1.0 / double(gRtpm.frames);
    const double candidates = std::max(gRtpm.candidates, 1.0);
    std::cout << "         rtpm: per frame " << gRtpm.candidates * inv << " deposits past the first hit, "
              << 100.0 * gRtpm.visible / candidates << "% in a marked cell"
              << (o.culling ? "" : " (Photon Culling off)") << "; stored "
              << gRtpm.stored[RTPM_CAUSTIC] * inv << " caustic and " << gRtpm.stored[RTPM_GLOBAL] * inv
              << " global photons\n"
              << "         Photon Culling " << (o.culling ? "on" : "off");
    if (o.culling)
        std::cout << " (mask of 2^" << o.cullBits << " cells)";
    std::cout << "; global photons rejected with probability " << o.reject
              << "; Stochastic Evaluation ";
    if (o.listSize > 0)
        std::cout << "of " << o.listSize << " photons per map";
    else
        std::cout << "off (every photon evaluated)";
    std::cout << "; caustic radius " << o.causticScale << " x the global one; "
              << (o.uniformKernel ? "uniform kernel" : "cone filter") << "; build "
              << rtpmBuildName(o.build) << "\n"
              << "         structures: mean " << gRtpm.accelBytes * inv / (1024.0 * 1024.0)
              << " MiB; device memory set aside: photons "
              << double(gRtpm.buffers.bytes) / (1024.0 * 1024.0) << " MiB, structures "
              << double(gRtpm.accelOutBytes[0] + gRtpm.accelOutBytes[1] + gRtpm.accelTempBytes)
            / (1024.0 * 1024.0)
              << " MiB (" << gRtpm.buffers.capacity[RTPM_CAUSTIC] << " + "
              << gRtpm.buffers.capacity[RTPM_GLOBAL] << " photon slots)\n";
    if (gRtpm.saturatedFrames > 0)
        std::cout << "         SATURATED: " << gRtpm.saturatedFrames
                  << " frames deposited more photons than fit; their estimate is short of them\n";
}

// The method's part of the timing record: its settings, and what a run did.
static void rtpmWriteJson(std::ostream& out)
{
    const RtpmOptions& o = gRtpm.options;
    const double inv = gRtpm.frames > 0 ? 1.0 / double(gRtpm.frames) : 0.0;
    out << "  \"rtpm\": {\"culling\": " << (o.culling ? "true" : "false")
        << ", \"mask_bits\": " << o.cullBits << ", \"reject\": " << o.reject
        << ", \"list_size\": " << o.listSize << ", \"caustic_scale\": " << o.causticScale
        << ", \"kernel\": \"" << (o.uniformKernel ? "uniform" : "cone") << "\""
        << ", \"build\": \"" << rtpmBuildName(o.build) << "\""
        << ", \"caustic_slots\": " << gRtpm.buffers.capacity[RTPM_CAUSTIC]
        << ", \"global_slots\": " << gRtpm.buffers.capacity[RTPM_GLOBAL]
        << ", \"photon_bytes\": " << gRtpm.buffers.bytes
        << ", \"structure_bytes_reserved\": "
        << (gRtpm.accelOutBytes[0] + gRtpm.accelOutBytes[1] + gRtpm.accelTempBytes)
        << ", \"structure_bytes_mean\": " << gRtpm.accelBytes * inv
        << ", \"candidates_mean\": " << gRtpm.candidates * inv
        << ", \"visible_mean\": " << gRtpm.visible * inv
        << ", \"caustic_mean\": " << gRtpm.stored[RTPM_CAUSTIC] * inv
        << ", \"global_mean\": " << gRtpm.stored[RTPM_GLOBAL] * inv
        << ", \"saturated_frames\": " << gRtpm.saturatedFrames << "},\n";
}

// The method's controls in the realtime GUI.
static bool rtpmGui(const FrameTimings& last)
{
    RtpmOptions& o = gRtpm.options;
    bool changed = false;
    changed |= ImGui::Checkbox("Photon Culling", &o.culling);
    changed |= ImGui::SliderFloat("Global photon rejection", &o.reject, 0.f, 0.95f, "%.2f");
    changed |= ImGui::SliderInt("Stochastic Evaluation list (0 = all)", &o.listSize, 0, RTPM_MAX_LIST);
    changed |= ImGui::SliderFloat("Caustic radius scale", &o.causticScale, 0.05f, 1.f, "%.2f");
    ImGui::Text("Stored %d of %lld deposits (%.1f%% outside marked cells)", last.photons,
        last.deposits, 100.0 * last.culled);
    return changed;
}
