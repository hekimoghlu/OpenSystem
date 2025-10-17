/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// aligned_memory_unittests: Tests for the aligned memory allocator.
//   Tests copied from Chrome: src/base/memory/aligned_memory_unittests.cc.
//

#include "common/aligned_memory.h"

#include <memory>

#include "gtest/gtest.h"

#define EXPECT_ALIGNED(ptr, align) EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(ptr) & (align - 1))

namespace angle
{

// Test that dynamic allocation works as expected.
TEST(AlignedMemoryTest, DynamicAllocation)
{
    void *p = AlignedAlloc(8, 8);
    EXPECT_TRUE(p);
    EXPECT_ALIGNED(p, 8);
    AlignedFree(p);

    p = AlignedAlloc(8, 16);
    EXPECT_TRUE(p);
    EXPECT_ALIGNED(p, 16);
    AlignedFree(p);

    p = AlignedAlloc(8, 256);
    EXPECT_TRUE(p);
    EXPECT_ALIGNED(p, 256);
    AlignedFree(p);

    p = AlignedAlloc(8, 4096);
    EXPECT_TRUE(p);
    EXPECT_ALIGNED(p, 4096);
    AlignedFree(p);
}

}  // namespace angle
