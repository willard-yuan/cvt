cmake_minimum_required (VERSION 2.6)
project (hnsw_lib)

SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

set(OpenCV_DIR /home/yuanyong/libs/opencv-3.4.0/share/OpenCV)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB hnsw_lib
    "*.h"
    "*.hpp"
    "*.cpp"
)

add_executable(main ${hnsw_lib})
target_link_libraries(main ${OpenCV_LIBS})
