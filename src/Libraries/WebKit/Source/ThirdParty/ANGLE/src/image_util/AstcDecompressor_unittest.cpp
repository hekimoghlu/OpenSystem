/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// AstcDecompressor_unittest.cpp: Unit tests for AstcDecompressor

#include <gmock/gmock.h>
#include <vector>

#include "common/WorkerThread.h"
#include "image_util/AstcDecompressor.h"
#include "image_util/AstcDecompressorTestUtils.h"

using namespace angle;
using namespace testing;

namespace
{

// Test that we can correctly decompress an image
TEST(AstcDecompressor, Decompress)
{
    const int width  = 1024;
    const int height = 1024;

    auto singleThreadedPool = WorkerThreadPool::Create(1, ANGLEPlatformCurrent());
    auto multiThreadedPool  = WorkerThreadPool::Create(0, ANGLEPlatformCurrent());

    auto &decompressor = AstcDecompressor::get();
    if (!decompressor.available())
        GTEST_SKIP() << "ASTC decompressor not available";

    std::vector<Rgba> output(width * height);
    std::vector<uint8_t> astcData = makeAstcCheckerboard(width, height);
    int32_t status =
        decompressor.decompress(singleThreadedPool, multiThreadedPool, width, height, 8, 8,
                                astcData.data(), astcData.size(), (uint8_t *)output.data());
    EXPECT_EQ(status, 0);

    std::vector<Rgba> expected = makeCheckerboard(width, height);

    ASSERT_THAT(output, ElementsAreArray(expected));
}

// Test that getStatusString returns non-null even for unknown statuses
TEST(AstcDecompressor, getStatusStringAlwaysNonNull)
{
    EXPECT_THAT(AstcDecompressor::get().getStatusString(-10000), NotNull());
}

}  // namespace