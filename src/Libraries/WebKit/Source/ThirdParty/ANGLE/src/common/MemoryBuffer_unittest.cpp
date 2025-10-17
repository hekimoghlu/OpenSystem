/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
// Copyright 2025 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Unit tests for ANGLE's MemoryBuffer class.
//

#include "common/MemoryBuffer.h"

#include <gtest/gtest.h>

using namespace angle;

namespace
{

// Test usage of MemoryBuffer with multiple resizes
TEST(MemoryBufferTest, MultipleResizes)
{
    MemoryBuffer buffer;

    ASSERT_TRUE(buffer.resize(100));
    ASSERT_EQ(buffer.size(), 100u);
    buffer.assertTotalAllocatedBytes(100u);
    buffer.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(buffer.resize(300));
    ASSERT_EQ(buffer.size(), 300u);
    buffer.assertTotalAllocatedBytes(400u);
    buffer.assertTotalCopiedBytes(100u);

    ASSERT_TRUE(buffer.resize(100));
    ASSERT_EQ(buffer.size(), 100u);
    buffer.assertTotalAllocatedBytes(400u);
    buffer.assertTotalCopiedBytes(100u);

    ASSERT_TRUE(buffer.resize(400));
    ASSERT_EQ(buffer.size(), 400u);
    buffer.assertTotalAllocatedBytes(800u);
    buffer.assertTotalCopiedBytes(200u);
}

// Test usage of MemoryBuffer with reserve and then multiple resizes
TEST(MemoryBufferTest, ReserveThenResize)
{
    MemoryBuffer buffer;

    ASSERT_TRUE(buffer.reserve(300));
    ASSERT_EQ(buffer.size(), 0u);

    ASSERT_TRUE(buffer.resize(100));
    ASSERT_EQ(buffer.size(), 100u);
    buffer.assertTotalAllocatedBytes(300u);
    buffer.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(buffer.resize(300));
    ASSERT_EQ(buffer.size(), 300u);
    buffer.assertTotalAllocatedBytes(300u);
    buffer.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(buffer.resize(100));
    ASSERT_EQ(buffer.size(), 100u);
    buffer.assertTotalAllocatedBytes(300u);
    buffer.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(buffer.resize(400));
    ASSERT_EQ(buffer.size(), 400u);
    buffer.assertTotalAllocatedBytes(700u);
    buffer.assertTotalCopiedBytes(100u);
}

// Test usage of MemoryBuffer with clearAndReserve and then multiple resizes
TEST(MemoryBufferTest, ClearAndReserveThenResize)
{
    MemoryBuffer buffer;

    ASSERT_TRUE(buffer.clearAndReserve(300));
    ASSERT_EQ(buffer.size(), 0u);

    ASSERT_TRUE(buffer.resize(100));
    ASSERT_EQ(buffer.size(), 100u);
    buffer.assertTotalAllocatedBytes(300u);
    buffer.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(buffer.resize(300));
    ASSERT_EQ(buffer.size(), 300u);
    buffer.assertTotalAllocatedBytes(300u);
    buffer.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(buffer.resize(100));
    ASSERT_EQ(buffer.size(), 100u);
    buffer.assertTotalAllocatedBytes(300u);
    buffer.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(buffer.clearAndReserve(400));
    ASSERT_EQ(buffer.size(), 0u);

    ASSERT_TRUE(buffer.resize(400));
    ASSERT_EQ(buffer.size(), 400u);
    buffer.assertTotalAllocatedBytes(700u);
    buffer.assertTotalCopiedBytes(0u);
}

// Test appending and destroying MemoryBuffer
TEST(MemoryBufferTest, AppendAndDestroy)
{
    MemoryBuffer bufferSrc;
    MemoryBuffer bufferDst;

    ASSERT_TRUE(bufferSrc.clearAndReserve(100));
    ASSERT_EQ(bufferSrc.size(), 0u);

    ASSERT_TRUE(bufferSrc.resize(100));
    ASSERT_EQ(bufferSrc.size(), 100u);
    bufferSrc.assertTotalAllocatedBytes(100u);
    bufferSrc.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(bufferDst.clearAndReserve(200));
    ASSERT_EQ(bufferDst.size(), 0u);

    ASSERT_TRUE(bufferDst.resize(100));
    ASSERT_EQ(bufferDst.size(), 100u);
    ASSERT_TRUE(bufferDst.append(bufferSrc));
    ASSERT_EQ(bufferDst.size(), 200u);
    bufferDst.assertTotalAllocatedBytes(200u);
    bufferDst.assertTotalCopiedBytes(0u);

    ASSERT_TRUE(bufferDst.append(bufferSrc));
    ASSERT_EQ(bufferDst.size(), 300u);
    bufferDst.assertTotalAllocatedBytes(500u);
    bufferDst.assertTotalCopiedBytes(200u);

    ASSERT_TRUE(bufferDst.append(bufferDst));
    ASSERT_EQ(bufferDst.size(), 600u);
    bufferDst.assertTotalAllocatedBytes(1100u);
    bufferDst.assertTotalCopiedBytes(500u);

    bufferDst.destroy();
    ASSERT_EQ(bufferDst.size(), 0u);
    bufferDst.assertTotalAllocatedBytes(0u);
    bufferDst.assertTotalCopiedBytes(0u);
}

}  // namespace
