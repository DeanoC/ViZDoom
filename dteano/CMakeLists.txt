SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

include(cuda.cmake)

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

file(GLOB SRCS *.cu *.h *.c *.cpp)
file(GLOB TEST_ONLY_SRCS *_test.cu *_test.h *_test.cc *_test.cpp)
file(GLOB MAIN_ONLY_SRCS main.cpp)

set(MAIN_SRCS ${SRCS})
list(REMOVE_ITEM MAIN_SRCS ${TEST_ONLY_SRCS})

set(TEST_SRCS ${SRCS})
list(REMOVE_ITEM TEST_SRCS ${MAIN_ONLY_SRCS})

cuda_add_executable(dteano ${MAIN_SRCS})
target_link_libraries(dteano ${VIZDOOM_LIBS} ${DTEANO_LINKER_LIBS})

cuda_add_executable(dteano_tests ${TEST_SRCS})
target_link_libraries(dteano_tests ${VIZDOOM_LIBS} ${DTEANO_LINKER_LIBS} gtest gtest_main)

