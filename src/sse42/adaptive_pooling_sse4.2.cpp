// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This file include sse42 specific optimization function. Compiling in clang.
// Note: 1, avoid use static class variable
//       2, do not depend function in other cpp
#include "adaptive_pooling_private.h"

template void EASY_JIT_EXPOSE poolAvgT<xsimd::sse4_2>(xsimd::sse4_2*, const float *srcData, float *dstData, int od, int oh, int ow, size_t spatIndOff,
    const size_t inStrides[5], size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW, const FuseConstAlgParamPrivate<float, xsimd::sse4_2> params, const FuseMutableParams& params_m);
