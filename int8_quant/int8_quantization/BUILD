cc_library(name = 'int8_quan',
  srcs = [
      'int8_quan.cc',
    ],
    cppflags = [
      "-DFINTEGER=int",
      "-fopenmp",
      "-m64",
      "-mpopcnt",
    ],
    deps = [
      "//third_party/faiss/BUILD:faiss",
      "//se/txt2vid_se/profile/proto/BUILD:photo_profile",
    ],
    link_all_symbols = True,
)

cc_test(name = 'in8_quan_test',
  srcs = [
      'in8_quan_test.cpp',
  ],
  deps = [
      ':int8_quan',
  ]
)

#cc_binary(name = 'sq_encode_decode',
#  srcs = [
#      'sq_encode_decode.cc',
#  ],
#  deps = [
#    "//third_party/faiss/BUILD:faiss",
#  ],
#)
