// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_public.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <random>

using namespace easy;

static void matmul_ref(const MatmulConstParam& params_c, const MatmulMutableParam& params_m) {
#define A(i, j) a[(j) + (i) * params_c.x_stride]
#define B(i, j) b[(j) + (i) * params_c.y_stride]
#define C(i, j) c[(j) + (i) * params_c.dst_stride]

    const auto *a = (float*)params_m.src_x;
    const auto *b = (float*)params_m.src_y;
    auto *c = (float*)params_m.dst_d;
    int i, j, p;
    for (i = 0; i < params_c.m; i++) {
        for (j = 0; j < params_c.n; j++) {
            C(i, j) = 0;
            for (p = 0; p < params_c.k; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

static void gen_random(float* a, const size_t size) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        a[i] = dist(mt);
    }
}

static void elewise(float* a, const size_t size, float val) {

    for (size_t i = 0; i < size; i++) {
        a[i] += val;
    }
}

TEST(Matmul, Func) {
#define M 6
#define K 1024
#define N 1000
#define ELE_VAL -0.2f
    MatmulConstParam params_c;
    params_c.m = M;
    params_c.n = N;
    params_c.k = K;
    params_c.x_stride = K;
    params_c.y_stride = N;
    params_c.dst_stride = N;
    FuseConstParams fuse_params_c;
    fuse_params_c.num = 1;
    fuse_params_c.types[0] = AlgType::Add_C;
    fuse_params_c.params[0].x1 = ELE_VAL;
    auto f = getMatmulFunc(params_c, fuse_params_c);
    printf("func addr = %p\n", f.getRawPointer());

    std::unique_ptr<float> a(new float[M * K]);
    std::unique_ptr<float> b(new float[K * N]);
    std::unique_ptr<float> d(new float[M * N]);
    std::unique_ptr<float> ref(new float[M * N]);
    gen_random(a.get(), M * K);
    gen_random(b.get(), K * N);

    MatmulMutableParam params_m;
    params_m.src_x = a.get();
    params_m.src_y = b.get();
    params_m.dst_d = ref.get();
    FuseMutableParams fuse_params_m;
    matmul_ref(params_c, params_m);
    elewise(ref.get(), M * N, ELE_VAL);
    params_m.dst_d = d.get();
    matmul(params_c, params_m, fuse_params_c, fuse_params_m);
    for (size_t i = 0; i < M * N; i++) {
        EXPECT_NEAR(d.get()[i], ref.get()[i], 0.0001f) << "matmul diff at index " << i;
    }
    
    f(params_m, fuse_params_m);
    for (size_t i = 0; i < M * N; i++) {
        EXPECT_NEAR(d.get()[i], ref.get()[i], 0.0001f) << "matmul(jit) diff at index " << i;
    }
}
