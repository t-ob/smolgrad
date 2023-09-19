file(GLOB SOURCES "*.cpp")

add_library(smolgrad ${SOURCES})

target_include_directories(smolgrad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
