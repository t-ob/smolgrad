# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.25)

# Project's name
project(gradlibTests)

# We built google test as a subproject, maybe it's a good idea to turn on this policy
# Check https://cmake.org/cmake/help/v3.10/policy/CMP0079.html for details
#cmake_policy(SET CMP0079 NEW)

# Compile as C++20
#set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
FetchContent_MakeAvailable(googletest)

# Find the Google Test package
#find_package(GTest REQUIRED)

# Add all your test cpp files
set(TEST_SRC
        test_Value.cpp
        test_Neuron.cpp
        test_Layer.cpp
        test_MultiLayerPerceptron.cpp
        # Add more test source files here
        main.cpp)

# Add the executable test
add_executable(runGradlibTests ${TEST_SRC})

# Link the test executable with the Google Test main entry point and your own test library
target_link_libraries(runGradlibTests gtest gtest_main gradlib)

target_include_directories(runGradlibTests PRIVATE ../src)

# Enable testing functionality
enable_testing()

# Add the tests
add_test(NAME runGradlibTests COMMAND runGradlibTests)
