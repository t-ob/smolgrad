# Collect all source files into a variable
file(GLOB SOURCES "*.cpp")

# Define a library target based on the source files
add_library(gradlib ${SOURCES})

# Specify include directories for the library
target_include_directories(gradlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
