cmake_minimum_required(VERSION 2.8)
project(brute_pro)

set(CMAKE_CXX_FLAGS "-std=c++11 -fno-operator-names")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


#include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB brute_pro
    "*.hpp"
    "*.cpp"
)

add_executable(brute_search ${brute_pro})
#target_link_libraries (brute_search ${OpenCV_LIBS})
target_link_libraries (brute_search)