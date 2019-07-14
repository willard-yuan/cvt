cmake_minimum_required(VERSION 3.7)
project(tf_hello)

set(CMAKE_CXX_STANDARD 11)

file(GLOB tf_pro
    "./src/*.hpp"
    "./src/*.cpp"
)

# OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# Eigen lib headers
include_directories("/usr/local/include/eigen3")

# TensorFlow headers
include_directories("/Users/willard/codes/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include")
include_directories("/Users/willard/codes/cpp/tf_build/tf/include")
include_directories("/Users/willard/codes/cpp/tf_build/tf/include/bazel-genfiles/")
include_directories("/Users/willard/codes/cpp/tf_build/tf/include/tensorflow/")
include_directories("/Users/willard/codes/cpp/tf_build/tf/include/third-party/")

add_executable(tf_hello ${tf_pro})

target_link_libraries(tf_hello ${OpenCV_LIBS} /usr/local/lib/libtensorflow_cc.so /usr/local/lib/libtensorflow_framework.so)
