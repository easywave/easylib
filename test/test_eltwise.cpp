// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_public.h"
#include "gtest/gtest.h"

using namespace easy;
TEST(Eltwise, Func) {
    RawBytes raw;
    EltwiseConstParam params_c;
    FuseConstParams fuse_params_c;
    auto f = getEltwiseFunc(raw, params_c, fuse_params_c);
    EXPECT_EQ(f.isValid(), true);
}