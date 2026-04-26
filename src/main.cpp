#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "params.h" // your own params struct

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
    OptixDeviceContext context;        // connection to the GPU
    OptixTraversableHandle gas_handle; // the BVH (scene acceleration structure)
    CUdeviceptr d_gas_output_buffer;   // GPU memory for the BVH
    CUdeviceptr d_vertices;            // GPU memory for vertex data    
    OptixModule ptx_module;            // compiled GPU code
    OptixPipelineCompileOptions pipeline_compile_options; // options for pipeline compilation   
    OptixPipeline pipeline;            // the full ray tracing pipeline
    OptixShaderBindingTable sbt;       // maps rays to shaders
    Params *d_params;                  // launch parameters on GPU
    OptixProgramGroup raygen_group;    // ray generation program group
    OptixProgramGroup miss_group;      // miss program group
    OptixProgramGroup hit_group;       // hit program group
};

// void createContext(RendererState &state) {
//     CUDA_CHECK(cudaFree(0));
//     OPTIX_CHECK(optixInit());
//     OptixDeviceContextOptions options = {};
//     OPTIX_CHECK(optixDeviceContextCreate(0, &options, &state.context));
// }

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
    // Copy vertices to GPU
    const std::vector<float3> vertices = {
        {-0.5f, -0.5f, 0.f},
        {0.5f, -0.5f, 0.f},
        {0.0f, 0.5f, 0.f}
    };

    const size_t vertices_size = vertices.size() * sizeof(float3);
    cudaMalloc(reinterpret_cast<void **>(&state.d_vertices), vertices_size);
    cudaMemcpy(
        reinterpret_cast<void *>(state.d_vertices),
        vertices.data(), vertices_size,
        cudaMemcpyHostToDevice
    );

    // Describe the geometry
    const int MAT_COUNT = 1; // one material = one white triangle
    uint32_t triangle_flags[MAT_COUNT] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices = (uint32_t)vertices.size();
    triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
    triangle_input.triangleArray.flags = triangle_flags;
    triangle_input.triangleArray.numSbtRecords = MAT_COUNT;

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
    state.pipeline_compile_options.numPayloadValues = 3; // data carried by each ray
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
    float3 color;
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

    // Miss record — set data BEFORE copying to GPU
    SbtRecord<MissData> ms_record = {};
    ms_record.data.bg_color = make_float3(1.f, 1.f, 1.f); // white background
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_group, &ms_record));
    CUdeviceptr d_ms_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ms_record), sizeof(ms_record)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_ms_record), &ms_record,
                          sizeof(ms_record), cudaMemcpyHostToDevice));

    // Hit record — set data BEFORE copying to GPU
    SbtRecord<HitData> hit_record = {};
    hit_record.data.color = make_float3(1.f, 0.5f, 0.f); // orange
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_group, &hit_record));
    CUdeviceptr d_hit_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hit_record), sizeof(hit_record)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hit_record), &hit_record,
                          sizeof(hit_record), cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = d_rg_record;
    state.sbt.missRecordBase = d_ms_record;
    state.sbt.missRecordStrideInBytes = (unsigned int)roundUp(sizeof(ms_record), OPTIX_SBT_RECORD_ALIGNMENT);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = d_hit_record;
    state.sbt.hitgroupRecordStrideInBytes = (unsigned int)roundUp(sizeof(hit_record), OPTIX_SBT_RECORD_ALIGNMENT);
    state.sbt.hitgroupRecordCount = 1;

    state.sbt.callablesRecordBase = 0;
    state.sbt.callablesRecordStrideInBytes = 0;
    state.sbt.callablesRecordCount = 0;

    std::cout << "sizeof RayGenRecord: " << sizeof(SbtRecord<RayGenData>) << "\n";
    std::cout << "sizeof MissRecord:   " << sizeof(SbtRecord<MissData>) << "\n";
    std::cout << "sizeof HitRecord:    " << sizeof(SbtRecord<HitData>) << "\n";
    std::cout << "OPTIX_SBT_RECORD_ALIGNMENT: " << OPTIX_SBT_RECORD_ALIGNMENT << "\n";
    std::cout << "OPTIX_SBT_RECORD_HEADER_SIZE: " << OPTIX_SBT_RECORD_HEADER_SIZE << "\n";
}

void launch(RendererState &state) {
    const int W = 512, H = 512;

    // Create an explicit CUDA stream
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUdeviceptr d_frame_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_frame_buffer), W * H * sizeof(uchar4)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_frame_buffer), 0, W * H * sizeof(uchar4)));

    Params params = {};
    params.frame_buffer = reinterpret_cast<uchar4 *>(d_frame_buffer);
    params.width = W;
    params.height = H;
    params.handle = state.gas_handle;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_params), &params,
                          sizeof(Params), cudaMemcpyHostToDevice));

    // Use the explicit stream here
    OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                            d_params, sizeof(Params),
                            &state.sbt,
                            W, H, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream)); // sync this stream specifically
    CUDA_CHECK(cudaGetLastError());

    std::vector<uchar4> pixels(W * H);
    CUDA_CHECK(cudaMemcpy(pixels.data(), reinterpret_cast<void *>(d_frame_buffer),
                          W * H * sizeof(uchar4), cudaMemcpyDeviceToHost));

    std::ofstream img("output.ppm");
    img << "P3\n"
        << W << " " << H << "\n255\n";
    for (auto &p : pixels)
        img << (int)p.x << " " << (int)p.y << " " << (int)p.z << "\n";

    std::cout << "Saved output.ppm" << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(reinterpret_cast<void *>(d_frame_buffer));
    cudaFree(reinterpret_cast<void *>(d_params));
}

int main() {
    RendererState state;

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