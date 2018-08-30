
cmake_minimum_required(VERSION 2.8.8)
project(dlfeat_pro)

#SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11" )
set(CMAKE_CXX_FLAGS "-std=c++11 -fno-operator-names")

set(Caffe_DIR /home/yuanyong/codes/caffe_bk/build/install/share/Caffe)
find_package(Caffe)

set(OpenCV_DIR /home/yuanyong/libs/opencv-3.4.0/share/OpenCV)
find_package(OpenCV REQUIRED)

set(OpenBLAS_DIR /home/yuanyong/libs/OpenBLAS-0.2.20/lib/cmake/openblas)
find_package(OpenBLAS REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "OpenBLAS library status:")
message(STATUS "    version: ${OpenBLAS_VERSION}")
message(STATUS "    include path: ${OpenBLAS_INCLUDE_DIRS}")
message(STATUS "    libraries: ${OpenBLAS_LIBRARIES}")


include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

file(GLOB dlfeat_pro
    "*.h"
    "*.cc"
    "*.cpp"
)

add_executable(dl_feature ${dlfeat_pro})
target_link_libraries(dl_feature ${Caffe_LIBRARIES})
