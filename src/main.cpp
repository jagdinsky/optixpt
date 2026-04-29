#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "params.h" // your own params struct

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                       \
    do                                                         \
    {                                                          \
        cudaError_t err = call;                                \
        if (err != cudaSuccess)                                \
        {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":"   \
                      << __LINE__ << " — "                     \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1);                                           \
        }                                                      \
    } while (0)

#define OPTIX_CHECK(call)                                       \
    do                                                          \
    {                                                           \
        OptixResult res = call;                                 \
        if (res != OPTIX_SUCCESS)                               \
        {                                                       \
            std::cerr << "OptiX error at " << __FILE__ << ":"   \
                      << __LINE__ << " — "                      \
                      << optixGetErrorString(res) << std::endl; \
            exit(1);                                            \
        }                                                       \
    } while (0)

struct RendererState {
    OptixDeviceContext context;                           // connection to the GPU
    OptixTraversableHandle gas_handle;                    // the BVH (scene acceleration structure)
    CUdeviceptr d_gas_output_buffer;                      // GPU memory for the BVH
    CUdeviceptr d_vertices;                               // GPU memory for vertex data
    OptixModule ptx_module;                               // compiled GPU code
    OptixPipelineCompileOptions pipeline_compile_options; // options for pipeline compilation   
    OptixPipeline pipeline;                               // the full ray tracing pipeline
    OptixShaderBindingTable sbt;                          // maps rays to shaders
    Params *d_params;                                     // launch parameters on GPU
    OptixProgramGroup raygen_group;                       // ray generation program group
    OptixProgramGroup miss_group;                         // miss program group
    OptixProgramGroup hit_group;                          // hit program group
    CUdeviceptr d_triangles;                              // GPU memory for triangle data
    CUdeviceptr d_materials;                              // GPU memory for material data
    uint32_t num_vertices;                                // number of vertices
    uint32_t num_materials;                               // number of materials
    std::vector<Material> host_materials;                 // host copy of materials for SBT population
    std::vector<uint32_t> sbt_offsets;                    // per-triangle mat_id, built at upload time
};

struct Scene
{
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
};

void loadScene(const std::string &basename, Scene &scene)
{
    const std::string scenes_dir = "../scenes/";
    const std::string obj_path = scenes_dir + basename + ".obj";
    const std::string mtl_dir = scenes_dir; // tinyobj looks here for the .mtl

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          obj_path.c_str(), mtl_dir.c_str()))
        throw std::runtime_error(warn + err);

    if (!warn.empty())
        std::cerr << "[tinyobj] " << warn << "\n";
    std::cout << "Loaded " << basename << ".obj — "
              << shapes.size() << " shape(s), "
              << materials.size() << " material(s)\n";

    // ── Triangles ──────────────────────────────────────────────────────────
    for (const auto &shape : shapes)
    {
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3)
        {
            auto idx0 = shape.mesh.indices[i + 0];
            auto idx1 = shape.mesh.indices[i + 1];
            auto idx2 = shape.mesh.indices[i + 2];

            Triangle tri = {};
            tri.v0 = make_float3(attrib.vertices[3 * idx0.vertex_index],
                                 attrib.vertices[3 * idx0.vertex_index + 1],
                                 attrib.vertices[3 * idx0.vertex_index + 2]);
            tri.v1 = make_float3(attrib.vertices[3 * idx1.vertex_index],
                                 attrib.vertices[3 * idx1.vertex_index + 1],
                                 attrib.vertices[3 * idx1.vertex_index + 2]);
            tri.v2 = make_float3(attrib.vertices[3 * idx2.vertex_index],
                                 attrib.vertices[3 * idx2.vertex_index + 1],
                                 attrib.vertices[3 * idx2.vertex_index + 2]);

            if (idx0.normal_index >= 0)
                tri.n0 = make_float3(attrib.normals[3 * idx0.normal_index],
                                     attrib.normals[3 * idx0.normal_index + 1],
                                     attrib.normals[3 * idx0.normal_index + 2]);
            if (idx1.normal_index >= 0)
                tri.n1 = make_float3(attrib.normals[3 * idx1.normal_index],
                                     attrib.normals[3 * idx1.normal_index + 1],
                                     attrib.normals[3 * idx1.normal_index + 2]);
            if (idx2.normal_index >= 0)
                tri.n2 = make_float3(attrib.normals[3 * idx2.normal_index],
                                     attrib.normals[3 * idx2.normal_index + 1],
                                     attrib.normals[3 * idx2.normal_index + 2]);

            tri.mat_id = std::max(0, shape.mesh.material_ids[i / 3]);
            scene.triangles.push_back(tri);
        }
    }

    // ── Materials ──────────────────────────────────────────────────────────
    for (const auto &mat : materials)
    {
        Material m;
        m.albedo = make_float3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        m.emission = make_float3(mat.emission[0], mat.emission[1], mat.emission[2]);
        scene.materials.push_back(m);
    }

    // Guarantee at least one material
    if (scene.materials.empty())
    {
        Material fallback;
        fallback.albedo = make_float3(0.6f, 0.4f, 0.2f);
        fallback.emission = make_float3(0.f, 0.f, 0.f);
        scene.materials.push_back(fallback);
        std::cerr << "[warn] No materials found — using fallback brown\n";
    }

    // Clamp all mat_ids to valid range
    int max_id = (int)scene.materials.size() - 1;
    for (auto &tri : scene.triangles)
        tri.mat_id = std::min(tri.mat_id, max_id);
}

void uploadBuffersToGPU(RendererState &state, const Scene &scene)
{
    size_t tri_bytes = scene.triangles.size() * sizeof(Triangle);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_triangles), tri_bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_triangles),
                          scene.triangles.data(), tri_bytes, cudaMemcpyHostToDevice));

    size_t mat_bytes = scene.materials.size() * sizeof(Material);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_materials), mat_bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_materials),
                          scene.materials.data(), mat_bytes, cudaMemcpyHostToDevice));

    std::vector<float3> vertices;
    vertices.reserve(scene.triangles.size() * 3);
    for (auto &tri : scene.triangles)
    {
        vertices.push_back(tri.v0);
        vertices.push_back(tri.v1);
        vertices.push_back(tri.v2);
    }
    size_t vert_bytes = vertices.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_vertices), vert_bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_vertices),
                          vertices.data(), vert_bytes, cudaMemcpyHostToDevice));
    state.num_vertices = (uint32_t)vertices.size();
    state.num_materials = (uint32_t)scene.materials.size();
    state.host_materials = scene.materials;
    state.sbt_offsets.reserve(scene.triangles.size());
    for (const auto &tri : scene.triangles)
        state.sbt_offsets.push_back((uint32_t)tri.mat_id);
}

void createContext(RendererState &state)
{
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL; // ← add this
    options.logCallbackFunction = [](unsigned int level, const char *tag,
                                     const char *message, void *)
    {
        std::cerr << "[OptiX][" << tag << "] " << message << "\n";
    };
    options.logCallbackLevel = 4; // verbose

    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &state.context));
}

void buildMeshAccel(RendererState& state) {
    const uint32_t MAT_COUNT = (uint32_t)std::max(1u, state.num_materials);
    std::vector<uint32_t> triangle_flags(MAT_COUNT, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

    // Upload SBT offset buffer
    CUdeviceptr d_sbt_offsets;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_sbt_offsets),
                          state.sbt_offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_sbt_offsets),
                          state.sbt_offsets.data(),
                          state.sbt_offsets.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices = state.num_vertices;
    triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
    triangle_input.triangleArray.flags = triangle_flags.data();
    triangle_input.triangleArray.numSbtRecords = MAT_COUNT;
    triangle_input.triangleArray.sbtIndexOffsetBuffer = d_sbt_offsets;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    // Ask OptiX how much memory is needed
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes;
    optixAccelComputeMemoryUsage(
        state.context, &accel_options,
        &triangle_input, 1, // 1 build input
        &sizes
    );

    // Allocate memory and build the BVH
    CUdeviceptr d_temp;
    cudaMalloc(reinterpret_cast<void **>(&d_temp), sizes.tempSizeInBytes);

    CUdeviceptr d_output;
    size_t compacted_size_offset = sizes.outputSizeInBytes;
    cudaMalloc(reinterpret_cast<void **>(&d_output),
               compacted_size_offset + 8); // +8 bytes to store the compacted size

    OptixAccelEmitDesc emit_desc = {};
    emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = d_output + compacted_size_offset;

    optixAccelBuild(
        state.context, 0,
        &accel_options,
        &triangle_input, 1,
        d_temp, sizes.tempSizeInBytes,
        d_output, sizes.outputSizeInBytes,
        &state.gas_handle,
        &emit_desc, 1
    );

    cudaFree(reinterpret_cast<void *>(d_temp)); // temp buffer no longer needed

    // Compact (shrink) the BVH
    size_t compacted_size;
    cudaMemcpy(
        &compacted_size,
        reinterpret_cast<void *>(emit_desc.result),
        sizeof(size_t), cudaMemcpyDeviceToHost
    );

    if (compacted_size < sizes.outputSizeInBytes) {
        cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer), compacted_size);
        optixAccelCompact(state.context, 0,
                          state.gas_handle,
                          state.d_gas_output_buffer,
                          compacted_size,
                          &state.gas_handle);
        cudaFree(reinterpret_cast<void *>(d_output));
    } else {
        state.d_gas_output_buffer = d_output;
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_sbt_offsets))); // no longer needed
}

void createModule(RendererState& state) {
    // Options for compiling the module
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    // Pipeline-wide options
    state.pipeline_compile_options = {};
    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 13; // data carried by each ray
    state.pipeline_compile_options.numAttributeValues = 2; // data carried from intersection
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params"; // matches device code

    state.pipeline_compile_options.usesPrimitiveTypeFlags =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    state.pipeline_compile_options.allowOpacityMicromaps = 0;

    // Load the compiled GPU binary (PTX) from disk
    std::ifstream ptx_file("device.ptx", std::ios::binary);
    std::string ptx_code(
        (std::istreambuf_iterator<char>(ptx_file)),
        std::istreambuf_iterator<char>()
    );

    // Create the module
    char log[2048];
    size_t log_size = sizeof(log);
    optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        ptx_code.c_str(),
        ptx_code.size(),
        log,
        &log_size,
        &state.ptx_module // output: the module handle
    );
}

void createProgramGroups(RendererState& state) {
    OptixProgramGroupOptions pg_options = {};
    char log[2048];
    size_t log_size = sizeof(log);

    // Raygen, firing an initial ray for each pixel
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = state.ptx_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__rg";
    optixProgramGroupCreate(
        state.context,
        &raygen_desc,
        1, // num program groups
        &pg_options,
        log, &log_size,
        &state.raygen_group
    );

    // Miss, called when a ray misses all geometry
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = state.ptx_module;
    miss_desc.miss.entryFunctionName = "__miss__ms";
    optixProgramGroupCreate(
        state.context,
        &miss_desc,
        1, // num program groups
        &pg_options,
        log, &log_size,
        &state.miss_group
    );

    // Hit, called when a ray hits geometry
    OptixProgramGroupDesc hit_desc = {};
    hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_desc.hitgroup.moduleCH = state.ptx_module;
    hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    optixProgramGroupCreate(
        state.context,
        &hit_desc,
        1, // num program groups
        &pg_options,
        log, &log_size,
        &state.hit_group
    );
}

void createPipeline(RendererState& state) {
    // Link the program groups into a pipeline
    OptixProgramGroup groups[] = {
        state.raygen_group,
        state.miss_group,
        state.hit_group
    };

    // Options for pipeline creation
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 2; // max ray bounce depth

    // Log buffer for pipeline creation
    char log[2048];
    size_t log_size = sizeof(log);

    OPTIX_CHECK( optixPipelineCreate(state.context,
                                    &state.pipeline_compile_options,
                                    &link_options,
                                    groups,
                                    3,
                                    log,
                                    &log_size,
                                    &state.pipeline)
    );

    // Required: set stack sizes or optixLaunch will fail with "Invalid value"
    OPTIX_CHECK( optixPipelineSetStackSize(
        state.pipeline,
        2048,   // direct callable stack size from traversal
        2048,   // direct callable stack size from state
        2048,   // continuation callable stack size
        1       // max traversable graph depth (1 = single GAS, no instancing)
    ) );
}

// One SBT record bundles a shader header + optional user data
template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData
{
}; // no extra data needed yet
struct MissData
{
    float3 bg_color;
};
struct HitData
{
    float3 albedo;   // diffuse color of the surface
    float3 emission; // for emissive/light surfaces
};

// Helper to round up to alignment
inline size_t roundUp(size_t val, size_t align)
{
    return ((val + align - 1) / align) * align;
}

void createSBT(RendererState &state) {
    state.sbt = {}; // zero everything including callablesRecordBase

    // Raygen record
    SbtRecord<RayGenData> rg_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_group, &rg_record));
    CUdeviceptr d_rg_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rg_record), sizeof(rg_record)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_rg_record), &rg_record,
                          sizeof(rg_record), cudaMemcpyHostToDevice));
    state.sbt.raygenRecord = d_rg_record;   

    // Miss record — set data BEFORE copying to GPU
    SbtRecord<MissData> ms_record = {};
    ms_record.data.bg_color = make_float3(1.f, 1.f, 1.f); // white background
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_group, &ms_record));
    CUdeviceptr d_ms_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ms_record), sizeof(ms_record)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_ms_record), &ms_record,
                          sizeof(ms_record), cudaMemcpyHostToDevice));
    state.sbt.missRecordBase = d_ms_record;
    state.sbt.missRecordStrideInBytes = (unsigned int)roundUp(sizeof(ms_record), OPTIX_SBT_RECORD_ALIGNMENT);
    state.sbt.missRecordCount = 1;

    // Hit record — set data BEFORE copying to GPU
    const uint32_t MAT_COUNT = std::max(1u, state.num_materials);
    std::vector<SbtRecord<HitData>> hit_records(MAT_COUNT);

    for (uint32_t i = 0; i < MAT_COUNT; ++i)
    {
        OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_group, &hit_records[i]));
        hit_records[i].data.albedo = state.host_materials[i].albedo;
        hit_records[i].data.emission = state.host_materials[i].emission;
    }

    size_t hit_bytes = MAT_COUNT * sizeof(SbtRecord<HitData>);
    CUdeviceptr d_hit_records;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hit_records), hit_bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hit_records),
                          hit_records.data(), hit_bytes, cudaMemcpyHostToDevice));

    state.sbt.hitgroupRecordBase = d_hit_records;
    state.sbt.hitgroupRecordStrideInBytes = (unsigned int)roundUp(sizeof(SbtRecord<HitData>),
                                                                  OPTIX_SBT_RECORD_ALIGNMENT);
    state.sbt.hitgroupRecordCount = MAT_COUNT;

    // No callables for now
    state.sbt.callablesRecordBase = 0;
    state.sbt.callablesRecordStrideInBytes = 0;
    state.sbt.callablesRecordCount = 0;

    std::cout << "sizeof RayGenRecord: " << sizeof(SbtRecord<RayGenData>) << "\n";
    std::cout << "sizeof MissRecord:   " << sizeof(SbtRecord<MissData>) << "\n";
    std::cout << "sizeof HitRecord:    " << sizeof(SbtRecord<HitData>) << "\n";
    std::cout << "OPTIX_SBT_RECORD_ALIGNMENT: " << OPTIX_SBT_RECORD_ALIGNMENT << "\n";
    std::cout << "OPTIX_SBT_RECORD_HEADER_SIZE: " << OPTIX_SBT_RECORD_HEADER_SIZE << "\n";
}

void launch(RendererState &state)
{
    const int W = 512, H = 512;
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUdeviceptr d_fb, d_accum;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_fb), W * H * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_accum), W * H * sizeof(float3)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_accum), 0, W * H * sizeof(float3)));

    Params p = {};
    p.frame_buffer = reinterpret_cast<uchar4 *>(d_fb);
    p.accum_buffer = reinterpret_cast<float3 *>(d_accum);
    p.width = W;
    p.height = H;
    p.handle = state.gas_handle;
    p.triangles = reinterpret_cast<Triangle *>(state.d_triangles);
    p.materials = reinterpret_cast<Material *>(state.d_materials);

    // Simple camera looking at the triangle (at Z=0) from Z=-2
    p.cam_eye = make_float3(0.f, 12.5f, -30.f); 
    p.cam_w = make_float3(0.f, 0.f, 1.f);
    p.cam_u = make_float3(1.f, 0.f, 0.f);
    p.cam_v = make_float3(0.f, -1.f, 0.f);

    p.samples_per_pixel = 64;
    p.max_depth = 8;
    p.frame_index = 0;

    CUdeviceptr d_p;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_p), sizeof(p)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_p), &p, sizeof(p), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(state.pipeline, stream, d_p, sizeof(p), &state.sbt, W, H, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    std::vector<uchar4> pixels(W * H);
    CUDA_CHECK(cudaMemcpy(pixels.data(), reinterpret_cast<void *>(d_fb),
                          W * H * sizeof(uchar4), cudaMemcpyDeviceToHost));

    std::ofstream img("output.ppm");
    img << "P3\n"
        << W << " " << H << "\n255\n";
    for (auto &px : pixels)
        img << (int)px.x << " " << (int)px.y << " " << (int)px.z << "\n";
    std::cout << "Saved output.ppm\n";

    cudaStreamDestroy(stream);
    cudaFree(reinterpret_cast<void *>(d_fb));
    cudaFree(reinterpret_cast<void *>(d_accum));
    cudaFree(reinterpret_cast<void *>(d_p));
}

int main() {
    RendererState state = {};

    Scene scene;

    loadScene("lowpoly_tree", scene);
    std::cout << "Materials loaded: " << scene.materials.size() << "\n";
    for (size_t i = 0; i < scene.materials.size(); ++i)
        std::cout << "  mat[" << i << "] albedo=("
                  << scene.materials[i].albedo.x << ", "
                  << scene.materials[i].albedo.y << ", "
                  << scene.materials[i].albedo.z << ")\n";

    uploadBuffersToGPU(state, scene);

    createContext(state);
    std::cout << "Context created." << std::endl;

    buildMeshAccel(state);
    std::cout << "BVH built." << std::endl;

    createModule(state);
    std::cout << "Module created." << std::endl;

    createProgramGroups(state);
    std::cout << "Program groups created." << std::endl;

    createPipeline(state);
    std::cout << "Pipeline created." << std::endl;

    createSBT(state);
    std::cout << "SBT created." << std::endl;

    // Validate everything before launch
    std::cout << "pipeline: " << state.pipeline << "\n";
    std::cout << "gas_handle: " << state.gas_handle << "\n";
    std::cout << "raygen SBT: " << state.sbt.raygenRecord << "\n";
    std::cout << "miss SBT: " << state.sbt.missRecordBase << "\n";
    std::cout << "hit SBT: " << state.sbt.hitgroupRecordBase << "\n";
    std::cout << "miss stride: " << state.sbt.missRecordStrideInBytes << "\n";
    std::cout << "hit stride: " << state.sbt.hitgroupRecordStrideInBytes << "\n";
    std::cout << "d_params: " << state.d_params << "\n";
    
    launch(state);

    return 0;
}