cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(XeTLA)

include(CTest)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/tools/cmake")
find_package(MKL REQUIRED)

# debug option
message(STATUS "'DEBUG' is set to " ${DEBUG})
if (${DEBUG})
add_compile_options(-debug=minimal -Rno-debug-disables-optimization -DDEBUG=${DEBUG})
endif ()

# log message print
set(LOG "off" CACHE STRING "on/off log message")
message(STATUS "log message is " ${LOG})
if (${LOG} STREQUAL "on")
    add_definitions(-DLOG_PRINT)
endif ()

add_compile_options(-fsycl -ffp-model=precise -Wall -Wextra)
add_link_options(-fsycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm)
link_libraries(-lgtest -lgtest_main)

# Examples and tests are not needed when another project uses the library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
add_subdirectory(tests)
add_subdirectory(examples)
endif()
