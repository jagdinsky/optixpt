#include "params.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

// GLEW must come before any GL header. GLFW_INCLUDE_NONE prevents GLFW from
// pulling in gl.h/glext.h on its own, which would conflict with GLEW.
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ── Error macros ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                                                       \
    do {                                                                                                       \
        cudaError_t e = (call);                                                                                \
        if (e != cudaSuccess) {                                                                                \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e) << "\n"; \
            exit(1);                                                                                           \
        }                                                                                                      \
    } while (0)

#define OPTIX_CHECK(call)                                                                                        \
    do {                                                                                                         \
        OptixResult r = (call);                                                                                  \
        if (r != OPTIX_SUCCESS) {                                                                                \
            std::cerr << "OptiX error " << __FILE__ << ":" << __LINE__ << " " << optixGetErrorString(r) << "\n"; \
            exit(1);                                                                                             \
        }                                                                                                        \
    } while (0)

#define GL_CHECK()                                                                       \
    do {                                                                                 \
        GLenum e = glGetError();                                                         \
        if (e != GL_NO_ERROR) {                                                          \
            std::cerr << "GL error " << __FILE__ << ":" << __LINE__ << " " << e << "\n"; \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// ── Host-side float3 math (device.cu operators are __device__ only) ──────────
inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float3 operator*(float t, float3 v) { return make_float3(t * v.x, t * v.y, t * v.z); }
inline float3 operator*(float3 v, float t) { return make_float3(t * v.x, t * v.y, t * v.z); }
inline float3& operator+=(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
inline float3& operator-=(float3& a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline float3 normalize(float3 v)
{
    float inv = 1.f / sqrtf(dot(v, v));
    return inv * v;
}

// ── Renderer state ───────────────────────────────────────────────────────────
struct RendererState {
    OptixDeviceContext context;
    OptixTraversableHandle gasHandle;
    CUdeviceptr dGasOutputBuffer;
    CUdeviceptr dVertices;
    OptixModule ptxModule;
    OptixPipelineCompileOptions pipelineCompileOptions;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    CUdeviceptr dParams;
    OptixProgramGroup raygenGroup;
    OptixProgramGroup missGroup;
    OptixProgramGroup hitGroup;
    CUdeviceptr dTriangles;
    CUdeviceptr dMaterials;
    uint32_t numVertices;
    uint32_t numMaterials;
    std::vector<Material> hostMaterials;
    std::vector<uint32_t> sbtOffsets;
};

struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
};

// ── glTF helpers ─────────────────────────────────────────────────────────────
static float3 gltfFloat3(const tinygltf::Model& model,
    const tinygltf::Accessor& acc, size_t i)
{
    const auto& bv = model.bufferViews[acc.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    size_t stride = acc.ByteStride(bv) ? acc.ByteStride(bv) : sizeof(float3);
    const float* p = reinterpret_cast<const float*>(
        buf.data.data() + bv.byteOffset + acc.byteOffset + i * stride);
    return make_float3(p[0], p[1], p[2]);
}

static uint32_t gltfIndex(const tinygltf::Model& model,
    const tinygltf::Accessor& acc, size_t i)
{
    const auto& bv = model.bufferViews[acc.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset;
    switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return base[i];
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return reinterpret_cast<const uint16_t*>(base)[i];
    default:
        return reinterpret_cast<const uint32_t*>(base)[i];
    }
}

// ── Scene loader ─────────────────────────────────────────────────────────────
void loadScene(const std::string& filename, Scene& scene)
{
    const std::string filepath = "../scenes/" + filename;
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn, err;
    bool ok = (filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".glb") == 0)
        ? loader.LoadBinaryFromFile(&model, &err, &warn, filepath)
        : loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
    if (!warn.empty())
        std::cerr << "[tinyGLTF warn] " << warn << "\n";
    if (!ok)
        throw std::runtime_error("[tinyGLTF error] " + err);

    for (const auto& gmat : model.materials) {
        Material m;
        const auto& pbr = gmat.pbrMetallicRoughness;
        float r = (float)pbr.baseColorFactor[0], g = (float)pbr.baseColorFactor[1], b = (float)pbr.baseColorFactor[2];
        if (r > 0.99f && g > 0.99f && b > 0.99f && pbr.baseColorTexture.index >= 0) {
            static const float3 dc[] = { { 0.80f, 0.35f, 0.15f }, { 0.70f, 0.30f, 0.10f },
                { 0.20f, 0.20f, 0.20f }, { 1.00f, 0.95f, 0.60f }, { 0.60f, 0.85f, 1.00f } };
            int idx = (int)(&gmat - &model.materials[0]);
            m.albedo = dc[std::min(idx, 4)];
        } else {
            m.albedo = make_float3(r, g, b);
        }
        m.emission = make_float3((float)gmat.emissiveFactor[0],
            (float)gmat.emissiveFactor[1],
            (float)gmat.emissiveFactor[2]);
        scene.materials.push_back(m);
    }
    if (scene.materials.empty()) {
        Material f;
        f.albedo = make_float3(0.6f, 0.4f, 0.2f);
        f.emission = make_float3(0, 0, 0);
        scene.materials.push_back(f);
    }
    const int maxMat = (int)scene.materials.size() - 1;

    auto flatNormal = [](float3 v0, float3 v1, float3 v2) {
        float3 e1 = v1 - v0, e2 = v2 - v0;
        float3 n = cross(e1, e2);
        float len = sqrtf(dot(n, n));
        return len > 0 ? (1.f / len) * n : make_float3(0, 1, 0);
    };

    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1)
                continue;
            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end())
                continue;
            const auto& posAcc = model.accessors[posIt->second];
            bool hasN = false;
            tinygltf::Accessor normAcc;
            auto normIt = prim.attributes.find("NORMAL");
            if (normIt != prim.attributes.end()) {
                normAcc = model.accessors[normIt->second];
                hasN = true;
            }
            int matId = (prim.material >= 0) ? std::min(prim.material, maxMat) : 0;

            auto makeTri = [&](uint32_t i0, uint32_t i1, uint32_t i2) {
                Triangle tri;
                tri.v0 = gltfFloat3(model, posAcc, i0);
                tri.v1 = gltfFloat3(model, posAcc, i1);
                tri.v2 = gltfFloat3(model, posAcc, i2);
                if (hasN) {
                    tri.n0 = gltfFloat3(model, normAcc, i0);
                    tri.n1 = gltfFloat3(model, normAcc, i1);
                    tri.n2 = gltfFloat3(model, normAcc, i2);
                } else {
                    tri.n0 = tri.n1 = tri.n2 = flatNormal(tri.v0, tri.v1, tri.v2);
                }
                tri.mat_id = matId;
                scene.triangles.push_back(tri);
            };

            if (prim.indices >= 0) {
                const auto& idxAcc = model.accessors[prim.indices];
                for (size_t t = 0; t < idxAcc.count / 3; ++t)
                    makeTri(gltfIndex(model, idxAcc, t * 3 + 0),
                        gltfIndex(model, idxAcc, t * 3 + 1),
                        gltfIndex(model, idxAcc, t * 3 + 2));
            } else {
                for (size_t t = 0; t < posAcc.count / 3; ++t)
                    makeTri((uint32_t)(t * 3), (uint32_t)(t * 3 + 1), (uint32_t)(t * 3 + 2));
            }
        }
    }
    std::cout << "Loaded " << filename << "  tri=" << scene.triangles.size()
              << "  mat=" << scene.materials.size() << "\n";
}

// ── GPU uploads ──────────────────────────────────────────────────────────────
void uploadBuffersToGPU(RendererState& state, const Scene& scene)
{
    size_t tb = scene.triangles.size() * sizeof(Triangle);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dTriangles), tb));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dTriangles), scene.triangles.data(), tb, cudaMemcpyHostToDevice));
    size_t mb = scene.materials.size() * sizeof(Material);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dMaterials), mb));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dMaterials), scene.materials.data(), mb, cudaMemcpyHostToDevice));
    std::vector<float3> verts;
    verts.reserve(scene.triangles.size() * 3);
    for (const auto& t : scene.triangles) {
        verts.push_back(t.v0);
        verts.push_back(t.v1);
        verts.push_back(t.v2);
    }
    size_t vb = verts.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dVertices), vb));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dVertices), verts.data(), vb, cudaMemcpyHostToDevice));
    state.numVertices = (uint32_t)verts.size();
    state.numMaterials = (uint32_t)scene.materials.size();
    state.hostMaterials = scene.materials;
    state.sbtOffsets.reserve(scene.triangles.size());
    for (const auto& t : scene.triangles)
        state.sbtOffsets.push_back((uint32_t)t.mat_id);
}

// ── OptiX setup (unchanged logic) ────────────────────────────────────────────
void createContext(RendererState& state)
{
    // On Optimus laptops the NVIDIA GPU is always CUDA device 0.
    // Force it before cudaFree(0) so CUDA initialises on the same device
    // that owns the GL context (which initGL() already forced to NVIDIA).
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions opt = {};
    opt.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    opt.logCallbackFunction = [](unsigned int, const char*, const char* msg, void*) { std::cerr << "[OptiX] " << msg << "\n"; };
    opt.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(0, &opt, &state.context));
}

void buildMeshAccel(RendererState& state)
{
    const uint32_t MC = std::max(1u, state.numMaterials);
    std::vector<uint32_t> flags(MC, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    CUdeviceptr dSbt;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbt), state.sbtOffsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbt), state.sbtOffsets.data(), state.sbtOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    OptixBuildInput ti = {};
    ti.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    ti.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    ti.triangleArray.vertexStrideInBytes = sizeof(float3);
    ti.triangleArray.numVertices = state.numVertices;
    ti.triangleArray.vertexBuffers = &state.dVertices;
    ti.triangleArray.flags = flags.data();
    ti.triangleArray.numSbtRecords = MC;
    ti.triangleArray.sbtIndexOffsetBuffer = dSbt;
    ti.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    ti.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sz;
    optixAccelComputeMemoryUsage(state.context, &ao, &ti, 1, &sz);
    CUdeviceptr dTmp, dOut;
    size_t cso = sz.outputSizeInBytes;
    cudaMalloc(reinterpret_cast<void**>(&dTmp), sz.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&dOut), cso + 8);
    OptixAccelEmitDesc ed;
    ed.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    ed.result = dOut + cso;
    optixAccelBuild(state.context, 0, &ao, &ti, 1, dTmp, sz.tempSizeInBytes, dOut, sz.outputSizeInBytes, &state.gasHandle, &ed, 1);
    cudaFree(reinterpret_cast<void*>(dTmp));
    size_t cs;
    cudaMemcpy(&cs, reinterpret_cast<void*>(ed.result), sizeof(size_t), cudaMemcpyDeviceToHost);
    if (cs < sz.outputSizeInBytes) {
        cudaMalloc(reinterpret_cast<void**>(&state.dGasOutputBuffer), cs);
        optixAccelCompact(state.context, 0, state.gasHandle, state.dGasOutputBuffer, cs, &state.gasHandle);
        cudaFree(reinterpret_cast<void*>(dOut));
    } else {
        state.dGasOutputBuffer = dOut;
    }
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dSbt)));
}

void createModule(RendererState& state)
{
    OptixModuleCompileOptions mco = {};
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    state.pipelineCompileOptions = {};
    state.pipelineCompileOptions.usesMotionBlur = false;
    state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipelineCompileOptions.numPayloadValues = 13;
    state.pipelineCompileOptions.numAttributeValues = 2;
    state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    state.pipelineCompileOptions.usesPrimitiveTypeFlags = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    state.pipelineCompileOptions.allowOpacityMicromaps = 0;
    std::ifstream f("device.ptx", std::ios::binary);
    std::string ptx(std::istreambuf_iterator<char>(f), {});
    char log[2048];
    size_t ls = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(state.context, &mco, &state.pipelineCompileOptions, ptx.c_str(), ptx.size(), log, &ls, &state.ptxModule));
}

void createProgramGroups(RendererState& state)
{
    OptixProgramGroupOptions pgo = {};
    char log[2048];
    size_t ls = sizeof(log);
    OptixProgramGroupDesc rd = {};
    rd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rd.raygen.module = state.ptxModule;
    rd.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &rd, 1, &pgo, log, &ls, &state.raygenGroup));
    OptixProgramGroupDesc md = {};
    md.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    md.miss.module = state.ptxModule;
    md.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &md, 1, &pgo, log, &ls, &state.missGroup));
    OptixProgramGroupDesc hd = {};
    hd.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hd.hitgroup.moduleCH = state.ptxModule;
    hd.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &hd, 1, &pgo, log, &ls, &state.hitGroup));
}

void createPipeline(RendererState& state)
{
    OptixProgramGroup groups[] = { state.raygenGroup, state.missGroup, state.hitGroup };
    OptixPipelineLinkOptions lo = {};
    lo.maxTraceDepth = 2;
    char log[2048];
    size_t ls = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(state.context, &state.pipelineCompileOptions, &lo, groups, 3, log, &ls, &state.pipeline));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, 2048, 2048, 2048, 1));
}

template <typename T>
struct SbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
struct RayGenData { };
struct MissData {
    float3 bgColor;
};
struct HitData {
    float3 albedo;
    float3 emission;
};
inline size_t roundUp(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

void createSBT(RendererState& state)
{
    state.sbt = {};
    SbtRecord<RayGenData> rg;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenGroup, &rg));
    CUdeviceptr drg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&drg), sizeof(rg)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(drg), &rg, sizeof(rg), cudaMemcpyHostToDevice));
    state.sbt.raygenRecord = drg;
    SbtRecord<MissData> ms;
    ms.data.bgColor = make_float3(1, 1, 1);
    OPTIX_CHECK(optixSbtRecordPackHeader(state.missGroup, &ms));
    CUdeviceptr dms;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dms), sizeof(ms)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dms), &ms, sizeof(ms), cudaMemcpyHostToDevice));
    state.sbt.missRecordBase = dms;
    state.sbt.missRecordStrideInBytes = (unsigned int)roundUp(sizeof(ms), OPTIX_SBT_RECORD_ALIGNMENT);
    state.sbt.missRecordCount = 1;
    const uint32_t MC = std::max(1u, state.numMaterials);
    std::vector<SbtRecord<HitData>> hrs(MC);
    for (uint32_t i = 0; i < MC; ++i) {
        OPTIX_CHECK(optixSbtRecordPackHeader(state.hitGroup, &hrs[i]));
        hrs[i].data.albedo = state.hostMaterials[i].albedo;
        hrs[i].data.emission = state.hostMaterials[i].emission;
    }
    size_t hb = MC * sizeof(SbtRecord<HitData>);
    CUdeviceptr dhr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dhr), hb));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dhr), hrs.data(), hb, cudaMemcpyHostToDevice));
    state.sbt.hitgroupRecordBase = dhr;
    state.sbt.hitgroupRecordStrideInBytes = (unsigned int)roundUp(sizeof(SbtRecord<HitData>), OPTIX_SBT_RECORD_ALIGNMENT);
    state.sbt.hitgroupRecordCount = MC;
    state.sbt.callablesRecordBase = 0;
    state.sbt.callablesRecordStrideInBytes = 0;
    state.sbt.callablesRecordCount = 0;
}

// ── OpenGL fullscreen quad ────────────────────────────────────────────────────
static GLuint gQuadVAO = 0, gQuadVBO = 0, gQuadProg = 0;

static void initQuad()
{
    const char* vs = "#version 330 core\n"
                     "layout(location=0) in vec2 pos;\n"
                     "out vec2 uv;\n"
                     "void main(){ uv=pos*0.5+0.5; gl_Position=vec4(pos,0,1); }\n";
    const char* fs = "#version 330 core\n"
                     "in vec2 uv; out vec4 col; uniform sampler2D tex;\n"
                     "void main(){ col=texture(tex,vec2(uv.x,1.0-uv.y)); }\n";
    auto compile = [](GLenum t, const char* src) {
        GLuint s = glCreateShader(t);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[512];
            glGetShaderInfoLog(s, 512, nullptr, buf);
            std::cerr << "Shader err: " << buf << "\n";
            exit(1);
        }
        return s;
    };
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    gQuadProg = glCreateProgram();
    glAttachShader(gQuadProg, v);
    glAttachShader(gQuadProg, f);
    glLinkProgram(gQuadProg);
    glDeleteShader(v);
    glDeleteShader(f);
    float quad[] = { -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1 };
    glGenVertexArrays(1, &gQuadVAO);
    glGenBuffers(1, &gQuadVBO);
    glBindVertexArray(gQuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);
}

static void drawQuad(GLuint tex)
{
    glUseProgram(gQuadProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(gQuadProg, "tex"), 0);
    glBindVertexArray(gQuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

// ── Camera input ─────────────────────────────────────────────────────────────
static double gLastX = 0, gLastY = 0;
static float gYaw = -90.f, gPitch = 0.f; // degrees
static bool gFirstMouse = true;
static bool gCameraChanged = true;

static void rebuildCameraVectors(Params& p)
{
    float yawR = gYaw * (float)M_PI / 180.f;
    float pitchR = gPitch * (float)M_PI / 180.f;
    float3 fwd = normalize(make_float3(cosf(pitchR) * cosf(yawR),
        sinf(pitchR),
        cosf(pitchR) * sinf(yawR)));
    float3 worldUp = make_float3(0, 1, 0);
    float3 right = normalize(cross(fwd, worldUp));
    float3 up = normalize(cross(right, fwd));
    p.cam_w = fwd;
    p.cam_u = right;
    p.cam_v = make_float3(-up.x, -up.y, -up.z); // down vector for screen-space
}

static void mouseCallback(GLFWwindow*, double xpos, double ypos)
{
    if (gFirstMouse) {
        gLastX = xpos;
        gLastY = ypos;
        gFirstMouse = false;
    }
    float dx = (float)(xpos - gLastX) * 0.15f;
    float dy = (float)(gLastY - ypos) * 0.15f;
    gLastX = xpos;
    gLastY = ypos;
    gYaw += dx;
    gPitch = std::max(-89.f, std::min(89.f, gPitch + dy));
    gCameraChanged = true;
}

static void handleKeys(GLFWwindow* win, Params& p, float speed)
{
    float3 fwd = p.cam_w;
    float3 right = p.cam_u;
    bool moved = false;
    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) {
        p.cam_eye += speed * fwd;
        moved = true;
    }
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) {
        p.cam_eye -= speed * fwd;
        moved = true;
    }
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) {
        p.cam_eye -= speed * right;
        moved = true;
    }
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) {
        p.cam_eye += speed * right;
        moved = true;
    }
    if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) {
        p.cam_eye.y -= speed;
        moved = true;
    }
    if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) {
        p.cam_eye.y += speed;
        moved = true;
    }
    if (moved)
        gCameraChanged = true;
}

// ── Main launch / render loop ─────────────────────────────────────────────────
GLFWwindow* initGL()
{
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        exit(1);
    }
    // Force discrete NVIDIA GPU on Optimus laptops.
    // Without this GLFW picks the Intel iGPU and CUDA-GL interop fails.
#ifdef _GLFW_X11
    // Works on X11; on Wayland use __NV_PRIME_RENDER_OFFLOAD=1 env var instead
#endif
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "OptiX Renderer", nullptr, nullptr);
    if (!window) {
        std::cerr << "GLFW window failed\n";
        exit(1);
    }
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        std::cerr << "GLEW init failed: " << glewGetErrorString(glewErr) << "\n";
        exit(1);
    }
    return window;
}

void launch(RendererState& state, GLFWwindow* window)
{
    const int W = 800, H = 600;

    initQuad();

    // ── PBO + CUDA interop ───────────────────────────────────────────────────
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    if (pbo == 0) {
        std::cerr << "glGenBuffers failed\n";
        exit(1);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H * sizeof(uchar4), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsResource* cudaPBO;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsWriteDiscard));

    GLuint displayTex;
    glGenTextures(1, &displayTex);
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // ── Accumulation buffer ──────────────────────────────────────────────────
    CUdeviceptr dAccum;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAccum), W * H * sizeof(float3)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dAccum), 0, W * H * sizeof(float3)));

    // ── Params ───────────────────────────────────────────────────────────────
    Params p = {};
    p.accum_buffer = reinterpret_cast<float3*>(dAccum);
    p.width = W;
    p.height = H;
    p.handle = state.gasHandle;
    p.triangles = reinterpret_cast<Triangle*>(state.dTriangles);
    p.materials = reinterpret_cast<Material*>(state.dMaterials);
    p.cam_eye = make_float3(-0.49f, 1.5f, -17.0f);
    gYaw = -90.f;
    gPitch = 0.f;
    rebuildCameraVectors(p);
    p.samples_per_pixel = 2;
    p.max_depth = 8;
    p.frame_index = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dParams), sizeof(Params)));

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Render loop ──────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            break;

        handleKeys(window, p, 0.3f);
        if (gCameraChanged) {
            rebuildCameraVectors(p);
            p.frame_index = 0;
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dAccum), 0, W * H * sizeof(float3)));
            gCameraChanged = false;
        }

        // Map PBO -> CUDA pointer
        uchar4* devPtr;
        size_t sz;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPBO, stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devPtr), &sz, cudaPBO));

        p.frame_buffer = devPtr;
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams), &p, sizeof(p), cudaMemcpyHostToDevice, stream));
        OPTIX_CHECK(optixLaunch(state.pipeline, stream, state.dParams, sizeof(Params), &state.sbt, W, H, 1));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPBO, stream));

        p.frame_index++;

        // PBO -> texture -> screen
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        drawQuad(displayTex);
        glfwSwapBuffers(window);
    }

    cudaStreamDestroy(stream);
    cudaFree(reinterpret_cast<void*>(dAccum));
    cudaFree(reinterpret_cast<void*>(state.dParams));
    cudaGraphicsUnregisterResource(cudaPBO);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &displayTex);
    glfwDestroyWindow(window);
    glfwTerminate();
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::string sceneFile = (argc > 1) ? argv[1] : "scene.gltf";
    RendererState state;
    Scene scene;
    loadScene(sceneFile, scene);
    std::cout << "Materials: " << scene.materials.size() << "\n";
    // GL context must exist BEFORE CUDA initialises the device (cudaFree(0))
    // so that CUDA-GL interop works (cudaGraphicsGLRegisterBuffer).
    GLFWwindow* window = initGL();
    createContext(state);
    std::cout << "Context created.\n";
    uploadBuffersToGPU(state, scene);
    buildMeshAccel(state);
    std::cout << "BVH built.\n";
    createModule(state);
    std::cout << "Module created.\n";
    createProgramGroups(state);
    std::cout << "Program groups created.\n";
    createPipeline(state);
    std::cout << "Pipeline created.\n";
    createSBT(state);
    std::cout << "SBT created.\n";
    launch(state, window);
    return 0;
}