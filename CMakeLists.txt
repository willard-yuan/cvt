cmake_minimum_required(VERSION 2.8)
project(helloworld_SRC)
set(CMAKE_CXX_FLAGS "-std=c++11 -fno-operator-names")

if(IS_GNU OR IS_CLANG)
    if(NOT HAS_AVX_EXTENSION)
        add_definitions(-DVL_DISABLE_AVX)
    endif()

    if(NOT HAS_SSE2_EXTENSION)
        add_definitions(-DVL_DISABLE_SSE2)
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SSE_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS}")
else()
    add_definitions(-DVL_DISABLE_AVX)
    add_definitions(-DVL_DISABLE_SSE2)
endif()

if(NOT OPENMP_ENABLED OR NOT OPENMP_FOUND)
    add_definitions(-DVL_DISABLE_OPENMP)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OpenCV
set(OpenCV_DIR /usr/local/Cellar/opencv/3.3.0_3/share/OpenCV)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Armadillo
#find_package(Armadillo REQUIRED)
set(ARMADILLO_LIBRARIES "/usr/local/Cellar/armadillo/8.100.1/lib/libarmadillo.dylib")
set(ARMADILLO_INCLUDE_DIR "/usr/local/Cellar/armadillo/8.100.1/include")

# VLFeat
SET(VLFEAT_INCLUDE_DIRS "/Users/willard/Public/github/covdet/3rd/VLFeat/")
ADD_DEFINITIONS("-DVL_DISABLE_AVX")
FILE(GLOB VLFEAT_SOURCE_FILES /Users/willard/Public/github/covdet/3rd/VLFeat/vl/*.c) # PATH TO vlfeat so we can do ugly glob
MESSAGE(STATUS "VLFeat sources path: ${VLFEAT_SOURCE_FILES}")
ADD_LIBRARY(vlfeats ${VLFEAT_SOURCE_FILES} )

# Include
INCLUDE_DIRECTORIES(${VLFEAT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})

# Source
file(GLOB helloworld_SRC
    "*.hpp"
    "*.cpp"
    )

add_executable(main ${helloworld_SRC})
target_link_libraries (main ${OpenCV_LIBS} vlfeats ${ARMADILLO_LIBRARIES})
