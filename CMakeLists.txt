if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 3.19)

project(ddim_scheduler_cpp)

set(CMAKE_CXX_STANDARD 17)

option(DDIM_SHARED_LIB "shared library support" OFF)
option(DDIM_INSTALL_LIB "install ddim_scheduler_cpp library and headers" ON)
option(DDIM_EXAMPLE "build example" OFF)

include_directories(source)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty)

file(GLOB_RECURSE SOURCE_FILES source/*.cpp)

if(DDIM_SHARED_LIB)
    add_library(ddim_scheduler_cpp SHARED ${SOURCE_FILES})
else()
    add_library(ddim_scheduler_cpp STATIC ${SOURCE_FILES})
endif()
set_target_properties(ddim_scheduler_cpp PROPERTIES DEBUG_POSTFIX "d")

if(DDIM_INSTALL_LIB)
    install(TARGETS ddim_scheduler_cpp EXPORT ddim_scheduler_cpp
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            )
    install(FILES ${PROJECT_SOURCE_DIR}/source/ddimscheduler.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/ddim_scheduler_cpp)

endif()

if(DDIM_EXAMPLE)
    add_subdirectory(example)
endif()
