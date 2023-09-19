cmake_minimum_required(VERSION 3.25)
project(smolgrad)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)

add_executable(mlp bin/mlp.cpp)
target_link_libraries(mlp smolgrad)