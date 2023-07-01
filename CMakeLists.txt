cmake_minimum_required(VERSION 3.25)
project(gradlib)

set(CMAKE_CXX_STANDARD 17)

#add_library(gradlib library.cpp include/Value.h src/Value.cpp)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)