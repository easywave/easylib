# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "easylib")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

file(GLOB PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/*.h)
file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB INSIDE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

if(NOT ENABLE_SS)
    file(GLOB SSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sse42/*.cpp)
    file(GLOB SSE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/sse42/*.h)

    list(APPEND INSIDE_HEADERS ${SSE_HEADERS})
    list(APPEND LIBRARY_SRC ${SSE_SRC})

    set_source_files_properties(${SSE_SRC} PROPERTIES COMPILE_OPTIONS "-msse4.2")
    add_definitions(-DHAVE_SSE=1)
endif()

if(NOT ENABLE_SS)
    file(GLOB AVX2_SRC ${CMAKE_CURRENT_SOURCE_DIR}/avx2/*.cpp)
    file(GLOB AVX2_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/avx2/*.h)

    list(APPEND INSIDE_HEADERS ${AVX2_HEADERS})
    list(APPEND LIBRARY_SRC ${AVX2_SRC})

    set_source_files_properties(${AVX2_SRC} PROPERTIES COMPILE_OPTIONS "-mavx2")
    add_definitions(-DHAVE_AVX2=1)
endif()

if(NOT ENABLE_SS)
    file(GLOB AVX512_SRC ${CMAKE_CURRENT_SOURCE_DIR}/avx512/*.cpp)
    file(GLOB AVX512_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/avx512/*.h)

    list(APPEND INSIDE_HEADERS ${AVX512_HEADERS})
    list(APPEND LIBRARY_SRC ${AVX512_SRC})

    set_source_files_properties(${AVX512_SRC} PROPERTIES COMPILE_OPTIONS "-mavx512f")
    add_definitions(-DHAVE_AVX512=1)
endif()

find_package(EasyJit REQUIRED CONFIG)
include_directories(${EasyJit_INCLUDE_DIRS})
link_directories(${EasyJit_LIBRARY_DIRS})

# https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Xclang -load -Xclang ${EasyJit_LIBRARY_DIRS}/EasyJitPass.so")

find_package(xsimd REQUIRED CONFIG)
include_directories(${xsimd_INCLUDE_DIRS})

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create library
# add_library(${TARGET_NAME}-objs
#     OBJECT
#     ${LIBRARY_SRC}
#     ${PUBLIC_HEADERS})
# add_library(${TARGET_NAME}-shared
#     SHARED
#     $<TARGET_OBJECTS:${TARGET_NAME}-objs>)
#add_library(${TARGET_NAME}-static
#    static
#    $<TARGET_OBJECTS:${TARGET_NAME}-objs>)
add_library(${TARGET_NAME}
    SHARED
    ${LIBRARY_SRC}
    ${INSIDE_HEADERS})
#set_target_properties(${TARGET_NAME}
#    PROPERTIES
#    PUBLIC_HEADER ${PUBLIC_HEADERS})
set_target_properties(${TARGET_NAME}
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib)
target_link_libraries(${TARGET_NAME} PRIVATE ${EasyJit_LIBRARY})

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR})

# install
install(TARGETS
            ${TARGET_NAME}
        RUNTIME DESTINATION
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/easylib/lib)
# public headers
INSTALL(
    DIRECTORY ${PUBLIC_HEADERS_DIR}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/easylib/include
    FILES_MATCHING PATTERN "*.h*")
# easyjit headers
INSTALL(
    DIRECTORY ${EasyJit_INCLUDE_DIRS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/easylib/include)
# easyjit runtime
FILE(GLOB esayjit_runtime "${EasyJit_LIBRARY_DIRS}/*${EasyJit_LIBRARY}*")
INSTALL(
    FILES ${esayjit_runtime}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/easylib/lib)