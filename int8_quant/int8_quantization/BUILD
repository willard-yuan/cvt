cc_library(name = 'int8_quan',
  srcs = [
      'int8_quan.cc',
    ],
    deps = [
      "//third_party/faiss/BUILD:faiss",
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
