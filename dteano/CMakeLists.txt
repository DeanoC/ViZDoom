#cmake_minimum_required( VERSION 2.4 )
#project(ViZDoomC++Examples)

#if(COMMAND cmake_policy)
#    cmake_policy(SET CMP0003 NEW)
#endif(COMMAND cmake_policy)
SET(USE_CUDNN)
include(cuda.cmake)

#add_library(vizdoom SHARED IMPORTED GLOBAL)
if (WIN32)
    set_property(TARGET vizdoom PROPERTY IMPORTED_LOCATION ../../bin/vizdoom.dll)
endif ()
if (UNIX AND NOT APPLE)
    set_property(TARGET vizdoom PROPERTY IMPORTED_LOCATION ../../bin/libvizdoom.so)
endif ()
if (APPLE)
    set_property(TARGET vizdoom PROPERTY IMPORTED_LOCATION ../../bin/libvizdoom.dylib)
endif ()

find_package(Boost COMPONENTS filesystem thread system date_time chrono regex iostreams REQUIRED)
find_package(Threads REQUIRED)

set(VIZDOOM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vizdoom_api_src)
include_directories(${VIZDOOM_INCLUDE_DIR} ${Boost_INCLUDE_DIR})

set(VIZDOOM_DTHEANO_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/dteano)

set(VIZDOOM_LIBS vizdoomstatic
        ${Boost_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT})

if (UNIX AND NOT APPLE)
    set(VIZDOOM_LIBS ${VIZDOOM_LIBS} rt)
endif ()

message(HAVE_CUDNN)

cuda_add_executable(dteano main.cpp cudainit.cu cudainit.h)
target_link_libraries(dteano ${VIZDOOM_LIBS} ${DTEANO_LINKER_LIBS})
