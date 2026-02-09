cmake_minimum_required(VERSION 3.16)

file(GLOB_RECURSE app_sources RELATIVE ${CMAKE_CURRENT_LIST_DIR} "src/*.c")

idf_component_register(SRCS ${app_sources} INCLUDE_DIRS "src" PRIV_INCLUDE_DIRS "." PRIV_REQUIRES heap)

# Enable Xtensa PIE optimizations for ESP32-S3
if(IDF_TARGET STREQUAL "esp32s3")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE CONFIG_IDF_TARGET_ESP32S3=1)
endif()

project(WebP)
