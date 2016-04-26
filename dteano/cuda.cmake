if (CPU_ONLY)
    return()
endif ()

################################################################################################
# Short command for cuDNN detection. Believe it soon will be a part of CUDA toolkit distribution.
# That's why not FindcuDNN.cmake file, but just the macro
# Usage:
#   detect_cuDNN()
function(detect_cuDNN)
    set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

    find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
            DOC "Path to cuDNN include directory.")

    get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
    find_library(CUDNN_LIBRARY NAMES libcudnn.so # libcudnn_static.a
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist}
            DOC "Path to cuDNN library.")

    if (CUDNN_INCLUDE AND CUDNN_LIBRARY)
        set(HAVE_CUDNN TRUE PARENT_SCOPE)
        set(CUDNN_FOUND TRUE PARENT_SCOPE)

        file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

        # cuDNN v3 and beyond
        string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
                CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
        string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
                CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
                CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
                CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

        if (NOT CUDNN_VERSION_MAJOR)
            set(CUDNN_VERSION "???")
        else ()
            set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
        endif ()

        message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

        string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 cuDNNVersionIncompatible)
        if (cuDNNVersionIncompatible)
            message(FATAL_ERROR "cuDNN version >3 is required.")
        endif ()

        set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
        mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

    endif ()
endfunction()

################################################################################################
###  Non macro section
################################################################################################

find_package(CUDA 7.5 QUIET)

if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found :(")
    return()
endif ()

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(HAVE_CUDA TRUE)
message(STATUS "CUDA detected: " ${CUDA_VERSION})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND DTEANO_LINKER_LIBS ${CUDA_CUDART_LIBRARY}
        ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})

add_definitions(-D__STRICT_ANSI__)
add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
add_definitions(-D_FORCE_INLINES)

# cudnn detection
if (USE_CUDNN)
    detect_cuDNN()
    if (HAVE_CUDNN)
        add_definitions(-DUSE_CUDNN)
        include_directories(SYSTEM ${CUDNN_INCLUDE})
        list(APPEND DTEANO_LINKER_LIBS ${CUDNN_LIBRARY})
    endif ()
endif ()

# setting nvcc arch flags
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE;")

# setting default testing device
if (NOT CUDA_TEST_DEVICE)
    set(CUDA_TEST_DEVICE -1)
endif ()

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)