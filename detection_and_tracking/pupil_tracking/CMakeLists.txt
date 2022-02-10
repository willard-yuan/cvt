cmake_minimum_required(VERSION 3.5)
project(pupil_tracking_pro)

set(CMAKE_C_FLAGS "-m64")
set(CMAKE_CXX_FLAGS "-std=c++11 -m64")
#set(LIB_DIR "/opt/local/lib")

set(CMAKE_CXX_FLAGS "-std=c++11 -fno-operator-names")

option(USE_SIMD_SSE2 "Enable SSE2 optimizations" OFF)
option(USE_SIMD_SSE3 "Enable SSE3 optimizations" OFF)
option(USE_SIMD_SSSE3 "Enable SSSE3 optimizations" OFF)
option(USE_SIMD_SSE4_1 "Enable SSE 4.1 optimizations" ON)
option(USE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" ON)
option(USE_SIMD_AVX "Enable AVX optimizations" ON)
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

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
set(CUDA_LIBRARY "/usr/local/cuda/lib64")

set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.4_4/lib/cmake/opencv4")
option( OpenCV_STATIC OFF )
find_package(OpenCV REQUIRED)

# Additional Library Directories
link_directories( ${OpenCV_LIB_DIR} )

# Set Static Link Runtime Library
if( OpenCV_STATIC )
  foreach( flag_var
           CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
           CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
           CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
           CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO )
    if( ${flag_var} MATCHES "/MD" )
      string( REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}" )
    endif()
  endforeach()
endif()

# OpenMP
find_package(OpenMP REQUIRED)
if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP_TARGET INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
    target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
    find_package(Threads REQUIRED)
    target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
    target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
endif()

#set(OpenCV_INCLUDE_DIRS "/raid/yuanyong/libs/opencv4.5.0/include/opencv4/")
#set(OpenCV_LIBS_DIRS "/raid/yuanyong/libs/opencv4.5.0/lib")
include_directories(${OpenCV_INCLUDE_DIRS})

# OpenCV message
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV config: ${OpenCV_DIR}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# OpenMP message
message(STATUS "OpenMP_VERSION: ${OpenMP_VERSION}")
message(STATUS "OpenMP_FOUND: ${OpenMP_FOUND}")
message(STATUS "OpenMP_CXX_FOUND: ${OpenMP_CXX_FOUND}")
message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
message(STATUS "OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")
message(STATUS "OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
message(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
message(STATUS "OpenMP_LIBRARY: ${OpenMP_LIBRARY}" )
message(STATUS "OpenMP_CXX_SPEC_DATE: ${OpenMP_CXX_SPEC_DATE}")
message(STATUS "OpenMP_CXX_VERSION_MAJOR: ${OpenMP_CXX_VERSION_MAJOR}")
message(STATUS "OpenMP_CXX_VERSION_MINOR: ${OpenMP_CXX_VERSION_MINOR}")
message(STATUS "OpenMP_CXX_VERSION: ${OpenMP_CXX_VERSION}")


#set(Faiss_INCLUDE_DIRS "/raid/yuanyong/libs/faiss-1.5.3-build/include")
#set(Faiss_LIBS_DIRS "/raid/yuanyong/libs/faiss-1.5.3-build/lib/libfaiss.so")
#include_directories(${Faiss_INCLUDE_DIRS})

file(GLOB pupil_tracking_files
    "./include/*.h"
    "./src/*.cc"
    "./examples/test.cc"
)

add_executable(pupil_tracking_demo ${pupil_tracking_files})
target_link_libraries(pupil_tracking_demo ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
