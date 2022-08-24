set(FOLDER_NAME "exe")

#if(IS_MSVC)
#    add_compile_options("/bigobj")
#endif()

COLMAP_ADD_EXECUTABLE(colmap_exe
    colmap.cc
    database.cc
    feature.cc
    #gui.cc
    #image.cc
    #model.cc
    #mvs.cc
    #sfm.cc
    vocab_tree.cc)
set_target_properties(colmap_exe PROPERTIES OUTPUT_NAME colmap)

COLMAP_ADD_SOURCES(
    feature.h feature.cc
    #sfm.h sfm.cc
)
