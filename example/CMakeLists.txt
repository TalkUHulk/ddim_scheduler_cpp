cmake_minimum_required(VERSION 3.19)

project(ddim_scheduler_cpp_example)

set(CMAKE_CXX_STANDARD 17)

set(DDIM_SCHEDULER_CPP_PATH ..)

add_executable(example example.cpp)

target_include_directories(example PRIVATE ${DDIM_SCHEDULER_CPP_PATH}/source)

target_link_libraries(example PRIVATE ddim_scheduler_cpp)