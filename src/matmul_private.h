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

#define A(i, j) a[(j) + (i) * params_c.k]
#define B(i, j) b[(j) + (i) * params_c.n]
#define C(i, j) c[(j) + (i) * params_c.n]

    const auto *a = (Type*)params_m.src_x;
    const auto *b = (Type*)params_m.src_y;
    auto *c = (Type*)params_m.dst_d;
    int i, j, p;
    memset(c, 0, params_c.m * params_c.n * sizeof(Type));

    Type *ptrC0, *ptrC1;
    b_t<Type, Arch> regCi0, regCi1;
    b_t<Type, Arch> regA0i0, regA0i1, regB0;
    b_t<Type, Arch> regA1i0, regA1i1, regB1;
    b_t<Type, Arch> regA2i0, regA2i1, regB2;
    b_t<Type, Arch> regA3i0, regA3i1, regB3;
    b_t<Type, Arch> regA4i0, regA4i1, regB4;
    b_t<Type, Arch> regA5i0, regA5i1, regB5;
    b_t<Type, Arch> regA6i0, regA6i1, regB6;
    b_t<Type, Arch> regA7i0, regA7i1, regB7;
    for (i = 0; i < params_c.m / 2 * 2; i += 2) {
        for (p = 0; p < params_c.k; p += 8) { // TODO: handle tail
            regA0i0 = b_t<Type, Arch>(A(i, p + 0));
            regA1i0 = b_t<Type, Arch>(A(i, p + 1));
            regA2i0 = b_t<Type, Arch>(A(i, p + 2));
            regA3i0 = b_t<Type, Arch>(A(i, p + 3));            
            regA4i0 = b_t<Type, Arch>(A(i, p + 4));
            regA5i0 = b_t<Type, Arch>(A(i, p + 5));
            regA6i0 = b_t<Type, Arch>(A(i, p + 6));
            regA7i0 = b_t<Type, Arch>(A(i, p + 7));    
            regA0i1 = b_t<Type, Arch>(A(i + 1, p + 0));
            regA1i1 = b_t<Type, Arch>(A(i + 1, p + 1));
            regA2i1 = b_t<Type, Arch>(A(i + 1, p + 2));
            regA3i1 = b_t<Type, Arch>(A(i + 1, p + 3));
            regA4i1 = b_t<Type, Arch>(A(i + 1, p + 4));
            regA5i1 = b_t<Type, Arch>(A(i + 1, p + 5));
            regA6i1 = b_t<Type, Arch>(A(i + 1, p + 6));
            regA7i1 = b_t<Type, Arch>(A(i + 1, p + 7));
            for (j = 0; j < params_c.n; j += inc) { // TODO: handle tail
                ptrC0 = &C(i, j);
                ptrC1 = &C(i + 1, j);
                regCi0 = b_t<Type, Arch>::load(ptrC0, xsimd::unaligned_mode());
                regCi1 = b_t<Type, Arch>::load(ptrC1, xsimd::unaligned_mode());
                regB0 = b_t<Type, Arch>::load(&B(p + 0, j), xsimd::unaligned_mode());
                regB1 = b_t<Type, Arch>::load(&B(p + 1, j), xsimd::unaligned_mode());
                regB2 = b_t<Type, Arch>::load(&B(p + 2, j), xsimd::unaligned_mode());
                regB3 = b_t<Type, Arch>::load(&B(p + 3, j), xsimd::unaligned_mode());
                regB4 = b_t<Type, Arch>::load(&B(p + 4, j), xsimd::unaligned_mode());
                regB5 = b_t<Type, Arch>::load(&B(p + 5, j), xsimd::unaligned_mode());
                regB6 = b_t<Type, Arch>::load(&B(p + 6, j), xsimd::unaligned_mode());
                regB7 = b_t<Type, Arch>::load(&B(p + 7, j), xsimd::unaligned_mode());
                regCi0 += regA0i0 * regB0 + regA1i0 * regB1 + 
                          regA2i0 * regB2 + regA3i0 * regB3 + 
                          regA4i0 * regB4 + regA5i0 * regB5 + 
                          regA6i0 * regB6 + regA7i0 * regB7;
                regCi1 += regA0i1 * regB0 + regA1i1 * regB1 + 
                          regA2i1 * regB2 + regA3i1 * regB3 + 
                          regA4i1 * regB4 + regA5i1 * regB5 + 
                          regA6i1 * regB6 + regA7i1 * regB7;
                regCi0.store_unaligned(ptrC0);
                regCi1.store_unaligned(ptrC1);
            }
        }
    }
    for (; i < params_c.m; i++) {
        for (p = 0; p < params_c.k; p += 8) { // TODO: handle tail
            regA0i0 = b_t<Type, Arch>(A(i, p + 0));
            regA1i0 = b_t<Type, Arch>(A(i, p + 1));
            regA2i0 = b_t<Type, Arch>(A(i, p + 2));
            regA3i0 = b_t<Type, Arch>(A(i, p + 3));            
            regA4i0 = b_t<Type, Arch>(A(i, p + 4));
            regA5i0 = b_t<Type, Arch>(A(i, p + 5));
            regA6i0 = b_t<Type, Arch>(A(i, p + 6));
            regA7i0 = b_t<Type, Arch>(A(i, p + 7));    
            for (j = 0; j < params_c.n; j += inc) { // TODO: handle tail
                ptrC0 = &C(i, j);
                regCi0 = b_t<Type, Arch>::load(ptrC0, xsimd::unaligned_mode());
                regB0 = b_t<Type, Arch>::load(&B(p + 0, j), xsimd::unaligned_mode());
                regB1 = b_t<Type, Arch>::load(&B(p + 1, j), xsimd::unaligned_mode());
                regB2 = b_t<Type, Arch>::load(&B(p + 2, j), xsimd::unaligned_mode());
                regB3 = b_t<Type, Arch>::load(&B(p + 3, j), xsimd::unaligned_mode());
                regB4 = b_t<Type, Arch>::load(&B(p + 4, j), xsimd::unaligned_mode());
                regB5 = b_t<Type, Arch>::load(&B(p + 5, j), xsimd::unaligned_mode());
                regB6 = b_t<Type, Arch>::load(&B(p + 6, j), xsimd::unaligned_mode());
                regB7 = b_t<Type, Arch>::load(&B(p + 7, j), xsimd::unaligned_mode());
                regCi0 += regA0i0 * regB0 + regA1i0 * regB1 + 
                          regA2i0 * regB2 + regA3i0 * regB3 + 
                          regA4i0 * regB4 + regA5i0 * regB5 + 
                          regA6i0 * regB6 + regA7i0 * regB7;
                regCi0.store_unaligned(ptrC0);
            }
        }
    }
}

template <class Type, class Arch>
void matmulT_(Arch*, const MatmulConstParam params_c, const MatmulMutableParam& params_m, const FuseConstAlgParamPrivate<Type, Arch> fuse_params_c, const FuseMutableParams& fuse_params_m) {
    constexpr std::size_t inc = b_t<Type, Arch>::size;

// #define A(i, j) a[(j) + (i) * params_c.n]
// #define B(i, j) b[(j) + (i) * params_c.k]
// #define C(i, j) c[(j) + (i) * params_c.k]

    const auto *a = (Type*)params_m.src_x;
    const auto *b = (Type*)params_m.src_y;
    const auto *c = (Type*)params_m.dst_d;
    int i, j, p;
    Type *ptrA0;
    Type *ptrA1;
    Type *ptrA2;
    Type *ptrA3;

    Type *ptrC0;
    Type *ptrC1;
    Type *ptrC2;
    Type *ptrC3;

    b_t<Type, Arch> regCj0;
    b_t<Type, Arch> regCj1;
    b_t<Type, Arch> regCj2;
    b_t<Type, Arch> regCj3;
    b_t<Type, Arch> regA0, regB0j0, regB0j1, regB0j2, regB0j3;
    b_t<Type, Arch> regA1, regB1j0, regB1j1, regB1j2, regB1j3;
    b_t<Type, Arch> regA2, regB2j0, regB2j1, regB2j2, regB2j3;
    b_t<Type, Arch> regA3, regB3j0, regB3j1, regB3j2, regB3j3;

    for (j = 0; j < params_c.n; j+=4) {   /* Loop over the columns of C */
        for (p = 0; p < params_c.k; p+=4) { /* Update C( i,j ) with the inner */
            regB0j0 = _mm_loaddup_pd(&B(p+0, j));
            regB1j0 = _mm_loaddup_pd(&B(p+1, j));
            regB2j0 = _mm_loaddup_pd(&B(p+2, j));
            regB3j0 = _mm_loaddup_pd(&B(p+3, j));

            regB0j1 = _mm_loaddup_pd(&B(p+0, j+1));
            regB1j1 = _mm_loaddup_pd(&B(p+1, j+1));
            regB2j1 = _mm_loaddup_pd(&B(p+2, j+1));
            regB3j1 = _mm_loaddup_pd(&B(p+3, j+1));

            regB0j2 = _mm_loaddup_pd(&B(p+0, j+2));
            regB1j2 = _mm_loaddup_pd(&B(p+1, j+2));
            regB2j2 = _mm_loaddup_pd(&B(p+2, j+2));
            regB3j2 = _mm_loaddup_pd(&B(p+3, j+2));

            regB0j3 = _mm_loaddup_pd(&B(p+0, j+3));
            regB1j3 = _mm_loaddup_pd(&B(p+1, j+3));
            regB2j3 = _mm_loaddup_pd(&B(p+2, j+3));
            regB3j3 = _mm_loaddup_pd(&B(p+3, j+3));

            for (i = 0; i < params_c.m; i += 2) { /* Loop over the rows of C */
                ptrC0 = &C(i, j);
                ptrC1 = &C(i, j + 1);
                ptrC2 = &C(i, j + 2);
                ptrC3 = &C(i, j + 3);

                ptrA0 = &A(i, p+0);
                ptrA1 = &A(i, p+1);
                ptrA2 = &A(i, p+2);
                ptrA3 = &A(i, p+3);

                regCj0 = _mm_load_pd(ptrC0);
                regCj1 = _mm_load_pd(ptrC1);
                regCj2 = _mm_load_pd(ptrC2);
                regCj3 = _mm_load_pd(ptrC3);

                regA0 = _mm_load_pd(ptrA0);
                regA1 = _mm_load_pd(ptrA1);
                regA2 = _mm_load_pd(ptrA2);
                regA3 = _mm_load_pd(ptrA3);

                regCj0 += regA0 * regB0j0 + regA1 * regB1j0;
                regCj0 += regA2 * regB2j0 + regA3 * regB3j0;

                regCj1 += regA0 * regB0j1 + regA1 * regB1j1;
                regCj1 += regA2 * regB2j1 + regA3 * regB3j1;

                regCj2 += regA0 * regB0j2 + regA1 * regB1j2;
                regCj2 += regA2 * regB2j2 + regA3 * regB3j2;

                regCj3 += regA0 * regB0j3 + regA1 * regB1j3;
                regCj3 += regA2 * regB2j3 + regA3 * regB3j3;

                _mm_store_pd(ptrC0, regCj0);
                _mm_store_pd(ptrC1, regCj1);
                _mm_store_pd(ptrC2, regCj2);
                _mm_store_pd(ptrC3, regCj3);
            }
        }
    }
}

}