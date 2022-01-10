// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// This file include common function needed by the optimization function
// Note: 1, Common function should be static to avoid linker pick the wrong architecture version
// 2, if want use specific instrisc can use the predefined macro: HAVE_AVX2/HAVE_SSE42/HAVE_AVX512
#include <math.h>
#include <vector>
#include <easy/jit.h>
#include "xsimd.h"
#include "fuse_private.h"
#include "eltwise_public.h"

namespace easy {
template <class Type, class Arch>
void eltwiseT(Arch*, const EltwiseConstParam params_c, const EltwiseMutableParam& params_m, const FuseConstAlgParamPrivate<Type, Arch> fuse_params_c, const FuseMutableParams& fuse_params_m) {
    constexpr std::size_t inc = b_t<Type, Arch>::size;

    const auto *src_x = (uint8_t*)params_m.src_x;
    const auto *src_y = (uint8_t*)params_m.src_y;
    const auto *dst_d = (uint8_t*)params_m.dst_d;
    int i = 0;
    for (; i < params_m.number / inc; i++) {
        // TODO: add type convert here
        auto x = b_t<Type, Arch>::load((Type*)(src_x + i * inc * sizeof(Type)), xsimd::unaligned_mode());
        auto y = b_t<Type, Arch>::load((Type*)(src_y + i * inc * sizeof(Type)), xsimd::unaligned_mode());
        b_t<Type, Arch> d;
        if (params_c.alg_type == AlgType::Add) {
            d = x + y;
        }
        // TODO: FIXME: check if need a new function to update the address in fuse params
        d = seq_fuse<Type, Arch>(d, i, fuse_params_m, fuse_params_c);
        // TODO: add type convert here
        d.store_unaligned((Type*)(dst_d + i * inc * sizeof(Type)));
    }
    if (params_m.number % inc) {
        Type buf[inc];
        // TODO: FIXME: malloc more data at least align simd width
        auto x = b_t<Type, Arch>::load((Type*)(src_x + i * inc * sizeof(Type)), xsimd::unaligned_mode());
        auto y = b_t<Type, Arch>::load((Type*)(src_y + i * inc * sizeof(Type)), xsimd::unaligned_mode());
        b_t<Type, Arch> d;
        if (params_c.alg_type == AlgType::Add) {
            d = x + y;
        }
        // TODO: FIXME: check if need a new function to update the address in fuse params
        d = seq_fuse<Type, Arch>(d, i, fuse_params_m, fuse_params_c);
        d.store_unaligned(buf);
        memcpy((void*)(dst_d + i * inc * sizeof(Type)), (void*)buf, (params_m.number % inc) * sizeof(Type));
    }
}

}