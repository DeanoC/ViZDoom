#cmake_minimum_required( VERSION 2.4 )
#project(ViZDoomC++Examples)

#if(COMMAND cmake_policy)
#    cmake_policy(SET CMP0003 NEW)
#endif(COMMAND cmake_policy)

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

add_executable(dteano main.cpp)
target_link_libraries(dteano ${VIZDOOM_LIBS})

#set_target_properties( dteano
#        PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
#        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
#        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
#        )
