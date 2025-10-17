/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// hash_utils_unittest: Hashing helper functions tests.

#include <gtest/gtest.h>

#include "common/hash_utils.h"

using namespace angle;

namespace
{
// Basic functionality test.
TEST(HashUtilsTest, ComputeGenericHash)
{
    std::string a = "aSimpleString!!!";
    std::string b = "anotherString???";

    // Requires a string size aligned to 4 bytes.
    ASSERT_TRUE(a.size() % 4 == 0);
    ASSERT_TRUE(b.size() % 4 == 0);

    size_t aHash = ComputeGenericHash(a.c_str(), a.size());
    size_t bHash = ComputeGenericHash(b.c_str(), b.size());

    EXPECT_NE(aHash, bHash);
}
}  // anonymous namespace
