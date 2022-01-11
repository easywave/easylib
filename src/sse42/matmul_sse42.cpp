// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_private.h"

namespace easy {
template void EASY_JIT_EXPOSE matmulT<float, xsimd::sse4_2>(xsimd::sse4_2*, const MatmulConstParam params_c, const MatmulMutableParam& params_m,
    const FuseConstAlgParamPrivate<float, xsimd::sse4_2> fuse_params_c, const FuseMutableParams& fuse_params_m);
}