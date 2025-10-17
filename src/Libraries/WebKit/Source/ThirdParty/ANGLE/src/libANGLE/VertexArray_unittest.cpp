/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
// Unit tests for VertexArray and related classes.
//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "common/bitset_utils.h"
#include "common/utilities.h"
#include "libANGLE/VertexArray.h"

using namespace gl;

// Tests that function GetIndexFromDirtyBit computes the index properly.
TEST(VertexArrayTest, VerifyGetIndexFromDirtyBit)
{
    VertexArray::DirtyBits dirtyBits;
    constexpr size_t bits[] = {2, 4, 9, 16, 25, 35};
    constexpr GLint count   = sizeof(bits) / sizeof(size_t);
    for (GLint i = 0; i < count; i++)
    {
        dirtyBits.set(bits[i]);
    }

    for (size_t dirtyBit : dirtyBits)
    {
        const size_t index = VertexArray::GetVertexIndexFromDirtyBit(dirtyBit);
        if (dirtyBit < VertexArray::DIRTY_BIT_ATTRIB_0)
        {
            continue;
        }
        else if (dirtyBit < VertexArray::DIRTY_BIT_ATTRIB_MAX)
        {
            EXPECT_EQ(dirtyBit - VertexArray::DIRTY_BIT_ATTRIB_0, index);
        }
        else if (dirtyBit < VertexArray::DIRTY_BIT_BINDING_MAX)
        {
            EXPECT_EQ(dirtyBit - VertexArray::DIRTY_BIT_BINDING_0, index);
        }
        else if (dirtyBit < VertexArray::DIRTY_BIT_BUFFER_DATA_MAX)
        {
            EXPECT_EQ(dirtyBit - VertexArray::DIRTY_BIT_BUFFER_DATA_0, index);
        }
        else
            ASSERT_TRUE(false);
    }
}
