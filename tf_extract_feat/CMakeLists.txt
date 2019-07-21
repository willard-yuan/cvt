cmake_minimum_required(VERSION 3.7)
project(tf_extract_feature)

set(CMAKE_CXX_STANDARD 11)

file(GLOB tf_pro
    "./include/*.hpp"
    "./src/*.cpp"
)

# OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# Eigen lib headers
include_directories("/usr/local/include/eigen3")

# TensorFlow headers
include_directories("/Users/willard/codes/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include")
include_directories("/Users/willard/libs/tf_1.9/include")
include_directories("/Users/willard/libs/tf_1.9/include/bazel-genfiles/")
include_directories("/Users/willard/libs/tf_1.9/include/tensorflow/")
include_directories("/Users/willard/libs/tf_1.9/include/third-party/")

add_executable(tf_extract_feature ${tf_pro})

target_link_libraries(tf_extract_feature ${OpenCV_LIBS} /usr/local/lib/libtensorflow_cc.so /usr/local/lib/libtensorflow_framework.so)
