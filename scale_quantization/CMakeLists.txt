cmake_minimum_required(VERSION 2.8)
project(scale_quantization_demo)

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

set(OpenCV_DIR "/home/yuanyong/libs/opencv-3.4.0/share/OpenCV")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(Faiss_INCLUDE_DIRS "/raid/yuanyong/libs/faiss-1.5.3-build/include")
set(Faiss_LIBS_DIRS "/raid/yuanyong/libs/faiss-1.5.3-build/lib/libfaiss.so")
include_directories(${Faiss_INCLUDE_DIRS})

file(GLOB sq_train_demo
    "./src/sq_train.cpp"
)
add_executable(sq_train_demo ${sq_train_demo})
target_link_libraries(sq_train_demo ${Faiss_LIBS_DIRS})

file(GLOB sq_encode_decode_demo
    "./src/sq_encode_decode.cpp"
)
add_executable(sq_encode_decode_demo ${sq_encode_decode_demo})
target_link_libraries(sq_encode_decode_demo ${Faiss_LIBS_DIRS})
