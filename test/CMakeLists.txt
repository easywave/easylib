# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "test")

find_package(EasyJit REQUIRED CONFIG)
include_directories(${EasyJit_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/../lib)

include_directories(${gtest_INCLUDE_DIRS})

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Create library
add_executable(${TARGET_NAME}
            ${SRC}
            ${HEADERS})
set_target_properties(${TARGET_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib)
target_link_libraries(${TARGET_NAME} PUBLIC easylib gtest gtest_main)

target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR})

# install
install(TARGETS 
          ${TARGET_NAME}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/easylib/lib
        RUNTIME DESTINATION
        LIBRARY DESTINATION)
