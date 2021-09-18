cc_library(
  name = "motion_area_detect_lib",
  srcs = [
    "src/canny_edge_detector.cc",
    "src/hough_transform.cc",
    "src/motion_area_detection.cc",
  ],
  cppflags = [
    "-Immu/similiar/vision",
    "-Wno-overloaded-virtual",
  ],
  deps = [
    "//mmu/online_service/protos/BUILD:video_edge_detection",
    "//third_party/gflags/BUILD:gflags",
    "//third_party/glog/BUILD:glog",
    "//third_party/OpenCV-3.4.0-CPU/BUILD:opencv",
  ],
)

cc_binary(
  name = "demo",
  srcs = [
    "demo.cc",
  ],
  cppflags = [
    "-Wno-overloaded-virtual",
  ],
  deps = [
    ":motion_area_detect_lib",
    "//third_party/protobuf_v3/BUILD:protoc_lib",
  ]
)
