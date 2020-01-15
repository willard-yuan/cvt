cmake_minimum_required(VERSION 2.8)
project(feature_demo)

set(CMAKE_C_FLAGS "-m64")
set(CMAKE_CXX_FLAGS "-std=c++11 -m64")
set(LIB_DIR "/opt/local/lib")

set(CMAKE_CXX_FLAGS "-std=c++11 -fno-operator-names")

option(USE_SIMD_SSE2 "Enable SSE2 optimizations" OFF)
option(USE_SIMD_SSE3 "Enable SSE3 optimizations" OFF)
option(USE_SIMD_SSSE3 "Enable SSSE3 optimizations" OFF)
option(USE_SIMD_SSE4_1 "Enable SSE 4.1 optimizations" OFF)
option(USE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" OFF)
option(USE_SIMD_AVX "Enable AVX optimizations" OFF)
option(USE_SIMD_AVX2 "Enable AVX2 optimizations" ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (USE_SIMD_SSE4_1)
    add_definitions(-DUSE_SSE)
    message(STATUS "Use SSE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()

if (USE_SIMD_AVX2)
    add_definitions(-DGLM_FORCE_PURE)
    if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        add_compile_options(-mavx2)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        add_compile_options(/QxAVX2)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/arch:AVX2)
    endif()
endif()

set(OpenCV_DIR "./libs/opencv-3.4.0/share/OpenCV")
find_package(OpenCV REQUIRED)


set(CMAKE_PREFIX_PATH "./libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(TORCH_INCLUDE_DIRS "./libtorch/include")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

file(GLOB feature_demo
    "./src/*.cpp"
    "./include/*.hpp"
)

add_executable(ares_feature ${feature_demo})
target_link_libraries(ares_feature ${OpenCV_LIBS} ${TORCH_LIBRARIES})
