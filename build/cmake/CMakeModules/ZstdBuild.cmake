# ################################################################
# ZSTD Build Targets Configuration
# ################################################################

# Always build the library first (this defines ZSTD_BUILD_STATIC/SHARED options)
add_subdirectory(lib)

# G9 : the GPU shared library (libzstd_gpu_shared.so) is placed in the lib
# build directory. Give targets created afterwards a build-tree RPATH so the
# --gpu binary runs without LD_LIBRARY_PATH.
if (ZSTD_BUILD_GPU)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)  # build tree uses the build RPATH
    set(CMAKE_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib")
endif()

# Validate build configuration after lib options are defined
if(ZSTD_BUILD_PROGRAMS)
    if(NOT ZSTD_BUILD_STATIC AND NOT ZSTD_PROGRAMS_LINK_SHARED)
        message(SEND_ERROR "Static library required to build zstd CLI programs")
    elseif(NOT ZSTD_BUILD_SHARED AND ZSTD_PROGRAMS_LINK_SHARED)
        message(SEND_ERROR "Shared library required to build zstd CLI programs")
    endif()
endif()

if(ZSTD_BUILD_TESTS AND NOT ZSTD_BUILD_STATIC)
    message(SEND_ERROR "Static library required to build test suite")
endif()

# Add programs if requested
if(ZSTD_BUILD_PROGRAMS)
    add_subdirectory(programs)
endif()

# Add tests if requested
if(ZSTD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Add contrib utilities if requested
if(ZSTD_BUILD_CONTRIB)
    add_subdirectory(contrib)
endif()

# Clean-all target for thorough cleanup
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/
    COMMENT "Performing complete clean including build directory"
)
