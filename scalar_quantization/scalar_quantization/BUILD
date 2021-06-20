cc_library(
    name = 'int8_quan',
    srcs = [
        'int8_quan.cc',
    ],
    cppflags = [
        '-DFINTEGER=int',
        '-fopenmp',
        '-m64',
        '-mpopcnt',
    ],
    deps = [
        '//third_party/nlohmann_json/BUILD:nlohmann_json',
        '//third_party/faiss/BUILD:faiss',
    ],
    link_all_symbols = True,
)

cc_test(
    name = 'int8_quan_test',
    srcs = [
        'int8_quan_test.cpp',
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
#    '//third_party/faiss/BUILD:faiss',
#  ],
#)
