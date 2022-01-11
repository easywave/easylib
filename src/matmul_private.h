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
#include "matmul_public.h"

namespace easy {
template <class Type, class Arch>
void matmulT(Arch*, const MatmulConstParam params_c, const MatmulMutableParam& params_m, const FuseConstAlgParamPrivate<Type, Arch> fuse_params_c, const FuseMutableParams& fuse_params_m) {
    constexpr std::size_t inc = b_t<Type, Arch>::size;

#define A(i, j) a[(j) + (i) * params_c.x_stride]
#define B(i, j) b[(j) + (i) * params_c.y_stride]
#define C(i, j) c[(j) + (i) * params_c.dst_stride]

    const auto *a = (Type*)params_m.src_x;
    const auto *b = (Type*)params_m.src_y;
    auto *c = (Type*)params_m.dst_d;
    int i, j, p;
    memset(c, 0, params_c.m * params_c.n * sizeof(Type));

    Type *ptrC0, *ptrC1, *ptrC2, *ptrC3;
    b_t<Type, Arch> regCi0, regCi1, regCi2, regCi3;
    b_t<Type, Arch> regA0i0, regA0i1, regA0i2, regA0i3, regB0;
    b_t<Type, Arch> regA1i0, regA1i1, regA1i2, regA1i3, regB1;
    b_t<Type, Arch> regA2i0, regA2i1, regA2i2, regA2i3, regB2;
    b_t<Type, Arch> regA3i0, regA3i1, regA3i2, regA3i3, regB3;

    for (i = 0; i < params_c.m / 4 * 4; i += 4) {
        for (p = 0; p < params_c.k; p += 4) { // TODO: handle tail
            regA0i0 = b_t<Type, Arch>(A(i, p + 0));
            regA1i0 = b_t<Type, Arch>(A(i, p + 1));
            regA2i0 = b_t<Type, Arch>(A(i, p + 2));
            regA3i0 = b_t<Type, Arch>(A(i, p + 3));
            regA0i1 = b_t<Type, Arch>(A(i + 1, p + 0));
            regA1i1 = b_t<Type, Arch>(A(i + 1, p + 1));
            regA2i1 = b_t<Type, Arch>(A(i + 1, p + 2));
            regA3i1 = b_t<Type, Arch>(A(i + 1, p + 3));
            regA0i2 = b_t<Type, Arch>(A(i + 2, p + 0));
            regA1i2 = b_t<Type, Arch>(A(i + 2, p + 1));
            regA2i2 = b_t<Type, Arch>(A(i + 2, p + 2));
            regA3i2 = b_t<Type, Arch>(A(i + 2, p + 3));
            regA0i3 = b_t<Type, Arch>(A(i + 3, p + 0));
            regA1i3 = b_t<Type, Arch>(A(i + 3, p + 1));
            regA2i3 = b_t<Type, Arch>(A(i + 3, p + 2));
            regA3i3 = b_t<Type, Arch>(A(i + 3, p + 3));
            for (j = 0; j < params_c.n; j += inc) { // TODO: handle tail
                ptrC0 = &C(i, j);
                ptrC1 = &C(i + 1, j);
                ptrC2 = &C(i + 2, j);
                ptrC3 = &C(i + 3, j);
                regCi0 = b_t<Type, Arch>::load(ptrC0, xsimd::unaligned_mode());
                regCi1 = b_t<Type, Arch>::load(ptrC1, xsimd::unaligned_mode());
                regCi2 = b_t<Type, Arch>::load(ptrC2, xsimd::unaligned_mode());
                regCi3 = b_t<Type, Arch>::load(ptrC3, xsimd::unaligned_mode());

                regB0 = b_t<Type, Arch>::load(&B(p + 0, j), xsimd::unaligned_mode());
                regB1 = b_t<Type, Arch>::load(&B(p + 1, j), xsimd::unaligned_mode());
                regB2 = b_t<Type, Arch>::load(&B(p + 2, j), xsimd::unaligned_mode());
                regB3 = b_t<Type, Arch>::load(&B(p + 3, j), xsimd::unaligned_mode());
                regCi0 += regA0i0 * regB0 + regA1i0 * regB1 + 
                          regA2i0 * regB2 + regA3i0 * regB3;
                regCi1 += regA0i1 * regB0 + regA1i1 * regB1 + 
                          regA2i1 * regB2 + regA3i1 * regB3;
                regCi2 += regA0i2 * regB0 + regA1i2 * regB1 + 
                          regA2i2 * regB2 + regA3i2 * regB3;
                regCi3 += regA0i3 * regB0 + regA1i3 * regB1 + 
                          regA2i3 * regB2 + regA3i3 * regB3;
                regCi0.store_unaligned(ptrC0);
                regCi1.store_unaligned(ptrC1);
                regCi2.store_unaligned(ptrC2);
                regCi3.store_unaligned(ptrC3);
            }
        }
    }
    for (; i < params_c.m; i++) {
        for (p = 0; p < params_c.k; p += 4) { // TODO: handle tail
            regA0i0 = b_t<Type, Arch>(A(i, p + 0));
            regA1i0 = b_t<Type, Arch>(A(i, p + 1));
            regA2i0 = b_t<Type, Arch>(A(i, p + 2));
            regA3i0 = b_t<Type, Arch>(A(i, p + 3));
            for (j = 0; j < params_c.n; j += inc) { // TODO: handle tail
                ptrC0 = &C(i, j);
                regCi0 = b_t<Type, Arch>::load(ptrC0, xsimd::unaligned_mode());
                regB0 = b_t<Type, Arch>::load(&B(p + 0, j), xsimd::unaligned_mode());
                regB1 = b_t<Type, Arch>::load(&B(p + 1, j), xsimd::unaligned_mode());
                regB2 = b_t<Type, Arch>::load(&B(p + 2, j), xsimd::unaligned_mode());
                regB3 = b_t<Type, Arch>::load(&B(p + 3, j), xsimd::unaligned_mode());
                regCi0 += regA0i0 * regB0 + regA1i0 * regB1 + 
                          regA2i0 * regB2 + regA3i0 * regB3;
                regCi0.store_unaligned(ptrC0);
            }
        }
    }

    // fuse operation
    const auto *dst_d = (uint8_t*)params_m.dst_d;
    i = 0;
    const auto size = params_c.m * params_c.n;
    for (; i < size / inc; i++) {
        // TODO: add type convert here
        auto x = b_t<Type, Arch>::load((Type*)(dst_d + i * inc * sizeof(Type)), xsimd::unaligned_mode());
        auto d = seq_fuse<Type, Arch>(x, i, fuse_params_m, fuse_params_c);
        // TODO: add type convert here
        d.store_unaligned((Type*)(dst_d + i * inc * sizeof(Type)));
    }
    if (size % inc) {
        Type buf[inc];
        // TODO: FIXME: malloc more data at least align simd width
        auto x = b_t<Type, Arch>::load((Type*)(dst_d + i * inc * sizeof(Type)), xsimd::unaligned_mode());
        auto d = seq_fuse<Type, Arch>(x, i, fuse_params_m, fuse_params_c);
        d.store_unaligned(buf);
        memcpy((void*)(dst_d + i * inc * sizeof(Type)), (void*)buf, (size % inc) * sizeof(Type));
    }
#undef A
#undef B
#undef C
}

}