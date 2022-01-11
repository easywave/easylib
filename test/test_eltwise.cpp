// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_public.h"
#include "gtest/gtest.h"
#include <algorithm>

using namespace easy;
TEST(Eltwise, Add) {
    EltwiseConstParam params_c;
    FuseConstParams fuse_params_c;
    fuse_params_c.num = 3;
    // first eltwise operation
    fuse_params_c.types[0] = AlgType::Add;
    fuse_params_c.types[1] = AlgType::Add_C;
    fuse_params_c.params[1].x1 = -1.1f;
    fuse_params_c.types[2] = AlgType::Add_C;
    fuse_params_c.params[2].x1 = 1.0f;
    auto f = getEltwiseFunc(params_c, fuse_params_c);
    EXPECT_EQ(f.isValid(), true);
    printf("func addr = %p\n", f.getRawPointer());
    
    std::array<float, 1600> x;
    std::array<float, 1600> d;
    x.fill(1.2f);
    EltwiseMutableParam params_m;
    params_m.src_x = x.data();
    params_m.dst_d = d.data();
    params_m.number = x.size();
    FuseMutableParams fuse_params_m;
    fuse_params_m.params[0].addr = reinterpret_cast<uint64_t>(x.data());
    f(params_m, fuse_params_m);

    std::array<float, 1600> d2;
    params_m.dst_d = d2.data();
    eltwise(params_c, params_m, fuse_params_c, fuse_params_m);
    EXPECT_EQ(memcmp(d.data(), d2.data(), d.size() * sizeof(float)), 0);
    EXPECT_EQ(std::all_of(d.begin(), d.end(), [](const float x) { return x == 1.2f + 1.2f - 1.1f + 1.0f; }), true);
}

TEST(Eltwise, ReLU) {
    EltwiseConstParam params_c;
    FuseConstParams fuse_params_c;
    fuse_params_c.num = 1;
    fuse_params_c.types[0] = AlgType::ReLU;
    auto f = getEltwiseFunc(params_c, fuse_params_c);
    printf("func addr = %p\n", f.getRawPointer());
    
    std::array<float, 1600> x;
    std::array<float, 1600> d;
    std::array<float, 1600> answer;
    int i = 0;
    std::for_each(x.begin(), x.end(), [&i, &answer](float& n) {
        if (i % 2 == 0) {
            n = -1.0f;
            answer[i] = 0;
        } else {
            n = 1.0f;
            answer[i] = 1.0f;
        }
        i++;
    });
    EltwiseMutableParam params_m;
    params_m.src_x = x.data();
    params_m.number = x.size();
    FuseMutableParams fuse_params_m;
    std::array<float, 1600> d2;
    params_m.dst_d = d2.data();
    eltwise(params_c, params_m, fuse_params_c, fuse_params_m);
    params_m.dst_d = d.data();
    f(params_m, fuse_params_m);

    EXPECT_EQ(memcmp(d.data(), d2.data(), d.size() * sizeof(float)), 0);
    EXPECT_EQ(memcmp(d.data(), answer.data(), d.size() * sizeof(float)), 0);
}

TEST(Eltwise, Mul) {
    EltwiseConstParam params_c;
    FuseConstParams fuse_params_c;
    fuse_params_c.num = 1;
    fuse_params_c.types[0] = AlgType::Mul_C;
    fuse_params_c.params[0].x1 = 2.5f;
    auto f = getEltwiseFunc(params_c, fuse_params_c);
    printf("func addr = %p\n", f.getRawPointer());
    
    std::array<float, 1600> x;
    std::array<float, 1600> d;
    std::array<float, 1600> answer;
    x.fill(10.3f);
    answer.fill(10.3f * 2.5f);

    EltwiseMutableParam params_m;
    params_m.src_x = x.data();
    params_m.number = x.size();
    FuseMutableParams fuse_params_m;
    std::array<float, 1600> d2;
    params_m.dst_d = d2.data();
    eltwise(params_c, params_m, fuse_params_c, fuse_params_m);
    params_m.dst_d = d.data();
    f(params_m, fuse_params_m);

    EXPECT_EQ(memcmp(d.data(), d2.data(), d.size() * sizeof(float)), 0);
    EXPECT_EQ(memcmp(d.data(), answer.data(), d.size() * sizeof(float)), 0);
}

TEST(Eltwise, BatchNormal) {
    EltwiseConstParam params_c;
    FuseConstParams fuse_params_c;
    fuse_params_c.num = 1;
    fuse_params_c.types[0] = AlgType::BatchNorm;
    fuse_params_c.params[0].x1 = 2.5f;
    fuse_params_c.params[0].x2 = 1.3f;
    auto f = getEltwiseFunc(params_c, fuse_params_c);
    printf("func addr = %p\n", f.getRawPointer());
    
    std::array<float, 1600> x;
    std::array<float, 1600> d;
    std::array<float, 1600> answer;
    x.fill(10.3f);
    answer.fill((10.3f - 2.5f) / 1.3f);

    EltwiseMutableParam params_m;
    params_m.src_x = x.data();
    params_m.number = x.size();
    FuseMutableParams fuse_params_m;
    std::array<float, 1600> d2;
    params_m.dst_d = d2.data();
    eltwise(params_c, params_m, fuse_params_c, fuse_params_m);
    params_m.dst_d = d.data();
    f(params_m, fuse_params_m);

    EXPECT_EQ(memcmp(d.data(), d2.data(), d.size() * sizeof(float)), 0);
    EXPECT_EQ(memcmp(d.data(), answer.data(), d.size() * sizeof(float)), 0);
}