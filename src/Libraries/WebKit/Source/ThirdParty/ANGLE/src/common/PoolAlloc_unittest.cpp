/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// PoolAlloc_unittest:
//   Tests of the PoolAlloc class
//

#include <gtest/gtest.h>

#include "common/PoolAlloc.h"

namespace angle
{
// Verify the public interface of PoolAllocator class
TEST(PoolAllocatorTest, Interface)
{
    size_t numBytes               = 1024;
    constexpr uint32_t kTestValue = 0xbaadbeef;
    // Create a default pool allocator and allocate from it
    PoolAllocator poolAllocator;
    void *allocation = poolAllocator.allocate(numBytes);
    // Verify non-zero ptr returned
    EXPECT_NE(nullptr, allocation);
    // Write to allocation to check later
    uint32_t *writePtr = static_cast<uint32_t *>(allocation);
    *writePtr          = kTestValue;
    // Test push and creating a new allocation
    poolAllocator.push();
    allocation = poolAllocator.allocate(numBytes);
    EXPECT_NE(nullptr, allocation);
    // Make an allocation that spans multiple pages
    allocation = poolAllocator.allocate(10 * 1024);
    // pop previous two allocations
    poolAllocator.pop();
    // Verify first allocation still has data
    EXPECT_EQ(kTestValue, *writePtr);
    // Make a bunch of allocations
    for (uint32_t i = 0; i < 1000; ++i)
    {
        numBytes   = (rand() % (1024 * 4)) + 1;
        allocation = poolAllocator.allocate(numBytes);
        EXPECT_NE(nullptr, allocation);
        // Write data into full allocation. In debug case if we
        //  overwrite any other allocation we get error.
        memset(allocation, 0xb8, numBytes);
    }
    // Free everything
    poolAllocator.popAll();
}

#if !defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
// Verify allocations are correctly aligned for different alignments
class PoolAllocatorAlignmentTest : public testing::TestWithParam<int>
{};

TEST_P(PoolAllocatorAlignmentTest, Alignment)
{
    int alignment = GetParam();
    // Create a pool allocator to allocate from
    PoolAllocator poolAllocator(4096, alignment);
    // Test a number of allocation sizes for each alignment
    for (uint32_t i = 0; i < 100; ++i)
    {
        // Vary the allocation size around 4k to hit some multi-page allocations
        const size_t numBytes = rand() % (1024 * 4) + 1;
        void *allocation      = poolAllocator.allocate(numBytes);
        // Verify alignment of allocation matches expected default
        EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(allocation) % alignment)
            << "Iteration " << i << " allocating " << numBytes;
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         PoolAllocatorAlignmentTest,
                         testing::Values(2, 4, 8, 16, 32, 64, 128),
                         testing::PrintToStringParamName());
#endif
}  // namespace angle
