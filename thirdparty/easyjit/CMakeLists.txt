# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Adds compiler flags to C / C++ sources
#
macro(add_compiler_flags)
    foreach(flag ${ARGN})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
    endforeach()
endmacro()

#set(BUILD_SHARED_LIBS OFF)
#set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(easyjit EXCLUDE_FROM_ALL)

get_target_property(easyjit_include_dirs easyjit INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(easyjit PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${easyjit_include_dirs}")

# set_target_properties(easyjit easyjit
#                       PROPERTIES FOLDER thirdparty)
