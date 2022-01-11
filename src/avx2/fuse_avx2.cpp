// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This file include avx2 specific optimization function. Compiling in clang.
// Note: 1, avoid use static class variable
//       2, do not depend function in other cpp
#include "fuse_private.h"

namespace easy {
template b_t<float, xsimd::avx2> EASY_JIT_EXPOSE add(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

template b_t<float, xsimd::avx2> EASY_JIT_EXPOSE sub(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

template b_t<float, xsimd::avx2> EASY_JIT_EXPOSE mul(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

template b_t<float, xsimd::avx2> EASY_JIT_EXPOSE relu(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

template b_t<float, xsimd::avx2> EASY_JIT_EXPOSE abs(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

template b_t<float, xsimd::avx2> EASY_JIT_EXPOSE add_c(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

template b_t<float, xsimd::avx2> EASY_JIT_EXPOSE sub_c(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

template b_t<float, xsimd::avx2> EASY_JIT_EXPOSE mul_c(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

}