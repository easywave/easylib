# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(gtest)

# easyjit
#execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory)
execute_process(COMMAND ${CMAKE_COMMAND}
    -S${CMAKE_SOURCE_DIR}/thirdparty/easyjit/easyjit
    -B${CMAKE_BINARY_DIR}/easyjit
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER=/usr/bin/clang-6.0
    -DCMAKE_C_COMPILER=/usr/bin/clang-6.0
    -DLLVM_DIR=/usr/lib/llvm-6.0/cmake
    -DEASY_JIT_EXAMPLE=OFF
    -DEASY_JIT_BENCHMARK=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/easyjit/)
execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/easyjit --target install --parallel)
#execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/easyjit/_install/lib/libEasyJitRuntime.so
#    ${CMAKE_BINARY_DIR}/../bin/intel64/${CMAKE_BUILD_TYPE}/lib/)

# xsimd
execute_process(COMMAND ${CMAKE_COMMAND}
    -S${CMAKE_SOURCE_DIR}/thirdparty/xsimd/xsimd
    -B${CMAKE_BINARY_DIR}/xsimd
    -DCMAKE_CXX_FLAGS="-DXSIMD_ENABLE_XTL_COMPLEX=0 "
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/xsimd)
execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/xsimd --target install)
