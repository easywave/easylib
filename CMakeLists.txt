# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(EasyLibs DESCRIPTION "easy library")

message(STATUS "Project will be installed to ${CMAKE_INSTALL_PREFIX}")

add_subdirectory(thirdparty)
set(xsimd_DIR ${CMAKE_INSTALL_PREFIX}/xsimd/lib/cmake/xsimd)
set(EasyJit_DIR ${CMAKE_INSTALL_PREFIX}/easyjit/lib/cmake)

add_subdirectory(src)

add_subdirectory(test)
