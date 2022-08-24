#[[if(IS_MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
elseif(IS_GNU OR IS_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    # We want c++14 mainly for supporting latest dependencies, but older GCC
    # versions in combination with CUDA do not yet support it. In the worst
    # case, this will lead to compile errors and there is not much we can
    # do on the COLMAP side.
    if(CUDA_ENABLED AND
       IS_GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    endif()
endif()]]

add_subdirectory(base)
#add_subdirectory(controllers)
add_subdirectory(estimators)
add_subdirectory(exe)
add_subdirectory(feature)
#add_subdirectory(mvs)
add_subdirectory(optim)
add_subdirectory(retrieval)
#add_subdirectory(sfm)
#add_subdirectory(tools)
add_subdirectory(util)
#if (GUI_ENABLED)
#    add_subdirectory(ui)
#endif()

COLMAP_ADD_STATIC_LIBRARY(colmap ${COLMAP_SOURCES})

if(CUDA_ENABLED)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")

    # Use a separate stream per thread to allow for concurrent kernel execution
    # between multiple threads on the same device.
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread")

    # Fix for Ubuntu 16.04.
    add_definitions("-D_MWAITXINTRIN_H_INCLUDED")

    if(MSVC)
        # Avoid pulling in too many header files through <windows.h>
        add_definitions("-DWIN32_LEAN_AND_MEAN")
    endif()

    COLMAP_ADD_STATIC_CUDA_LIBRARY(colmap_cuda ${COLMAP_CUDA_SOURCES})
    target_link_libraries(colmap_cuda colmap)

    target_link_libraries(colmap
        colmap_cuda ${COLMAP_INTERNAL_LIBRARIES} ${COLMAP_EXTERNAL_LIBRARIES})
else()
    target_link_libraries(colmap
        ${COLMAP_INTERNAL_LIBRARIES} ${COLMAP_EXTERNAL_LIBRARIES})
endif()
