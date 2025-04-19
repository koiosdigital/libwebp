cmake_minimum_required(VERSION 3.16)

file(GLOB_RECURSE app_sources RELATIVE ${CMAKE_CURRENT_LIST_DIR} "src/*.c")

idf_component_register(SRCS ${app_sources} INCLUDE_DIRS "src" PRIV_INCLUDE_DIRS "." PRIV_REQUIRES heap)

project(WebP)
