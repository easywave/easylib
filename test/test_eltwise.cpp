// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_public.h"
#include "gtest/gtest.h"

using namespace easy;
TEST(Eltwise, Add) {
    EltwiseConstParam params_c;
    params_c.alg_type = AlgType::Add;
    FuseConstParams fuse_params_c;
    fuse_params_c.num = 2;
    fuse_params_c.types[0] = AlgType::Add_C;
    fuse_params_c.params[0].x1 = 1.0f;
    fuse_params_c.types[1] = AlgType::Add_C;
    fuse_params_c.params[1].x1 = -1.1f;
    auto f = getEltwiseFunc(params_c, fuse_params_c);
    printf("func addr = %p\n", f.getRawPointer());
    
    std::array<float, 1600> x;
    std::array<float, 1600> d;
    x.fill(1.2f);
    EltwiseMutableParam params_m;
    params_m.src_x = x.data();
    params_m.src_y = x.data();
    params_m.dst_d = d.data();
    params_m.number = x.size();
    FuseMutableParams fuse_params_m;
    fuse_params_m.params[0].addr = 0;
    f(params_m, fuse_params_m);

    std::array<float, 1600> d2;
    params_m.dst_d = d2.data();
    eltwise(params_c, params_m, fuse_params_c, fuse_params_m);
    EXPECT_EQ(memcmp(d.data(), d2.data(), d.size() * sizeof(float)), 0);
    EXPECT_EQ(f.isValid(), true);
}