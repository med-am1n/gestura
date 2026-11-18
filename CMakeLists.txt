cmake_minimum_required(VERSION 3.10)
project(gestura)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)


set(IMGUI_SOURCES
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

add_executable(app
    src/main.cpp
    src/stb_image.cpp
    external/glad/src/glad.c
    ${IMGUI_SOURCES}
)

configure_file(
    ${CMAKE_SOURCE_DIR}/includes/root_directory.h.in
    ${CMAKE_BINARY_DIR}/generated/root_directory.h
)

find_library(ONNX_LIB onnxruntime HINTS "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime/lib" REQUIRED)

target_include_directories(app PRIVATE
    external/glad/include
    external/glm
    external/imgui
    includes/
    ${CMAKE_BINARY_DIR}/generated   
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime/include"
)


find_package(glfw3 REQUIRED)
find_package(assimp REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(app
    glfw
    assimp::assimp
    ${OpenCV_LIBS}
    ${ONNX_LIB}
    "-framework OpenGL"
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
)
