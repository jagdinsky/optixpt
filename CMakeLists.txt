cmake_minimum_required(VERSION 3.18)
project(renderer LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

set(OptiX_INCLUDE $ENV{HOME}/optix/include)

add_library(device_ptx OBJECT src/device.cu)
set_target_properties(device_ptx PROPERTIES CUDA_PTX_COMPILATION ON)
target_include_directories(device_ptx PRIVATE
    ${OptiX_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
)

add_library(imgui STATIC
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_tables.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_widgets.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/imgui_demo.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_SOURCE_DIR}/include/imgui/backends/imgui_impl_opengl3.cpp
)

add_library(miniz STATIC ${CMAKE_SOURCE_DIR}/include/miniz.c)
target_include_directories(miniz PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_include_directories(imgui PUBLIC
    ${CMAKE_SOURCE_DIR}/include/imgui
    ${CMAKE_SOURCE_DIR}/include/imgui/backends
)

target_link_libraries(imgui PUBLIC
    glfw
    OpenGL::GL
)

# main.cpp is host C++; cuda_passes.cu is compiled by nvcc into the same binary
# (it is where the grid build's CUB kernels live).  device.cu above is a
# different thing entirely — PTX for OptiX, never linked into the executable.
add_executable(renderer src/main.cpp src/cuda_passes.cu)

# "native" builds for the GPU in this machine.  Only the executable gets it:
# device_ptx has to stay a single-architecture PTX blob.
set_target_properties(renderer PROPERTIES CUDA_ARCHITECTURES native)

target_include_directories(renderer PRIVATE
    ${OptiX_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(renderer PRIVATE
    imgui
    miniz
    CUDA::cuda_driver
    CUDA::cudart
    OpenGL::GL
    glfw
    GLEW::GLEW
)

add_custom_command(TARGET renderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_OBJECTS:device_ptx>
        $<TARGET_FILE_DIR:renderer>/device.ptx
)