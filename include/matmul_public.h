// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fuse_public.h"
#include <easy/jit.h>

// reference: https://github.com/usstq/how-to-optimize-gemm
namespace easy {
// eltwise operation will sit in FuseConstParams.types[0]
struct MatmulMutableParam {
    void *src_x;    // src x addr
    void *src_y;

    void *dst_d;    // dst addr
};

struct MatmulConstParam {
    int m;
    int k;
    int n;
    int x_stride;
    int y_stride;
    int dst_stride;
    int precision;         // data precision
    MatmulConstParam() {
        memset(this, 0, sizeof(*this));
    }
};

using MatmulFunc = easy::FunctionWrapper<void(const MatmulMutableParam&, const FuseMutableParams&)>;

// raw optimization function
//  when debugging we can call this function directly
void matmul(const MatmulConstParam& params_c, const MatmulMutableParam& params_m, const FuseConstParams& fuse_params_c, const FuseMutableParams& fuse_params_m);

// get jit function, the parameters will become runtime constants
MatmulFunc getMatmulFunc(const MatmulConstParam& params_c, const FuseConstParams& fuse_params_c);

}