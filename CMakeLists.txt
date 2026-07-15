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

add_executable(renderer src/main.cpp)

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