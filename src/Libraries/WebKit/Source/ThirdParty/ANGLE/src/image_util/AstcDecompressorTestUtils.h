/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
// AstcDecompressorTestUtils.h: Utility functions for ASTC decompression tests

#include <vector>
#include "common/debug.h"

namespace testing
{
struct Rgba
{
    uint8_t r, g, b, a;
    bool operator==(const Rgba &o) const { return r == o.r && g == o.g && b == o.b && a == o.a; }
};
static_assert(sizeof(Rgba) == 4, "Rgba struct isn't 4 bytes");

// Creates a checkerboard image of a given size. The top left pixel will be black, and the remaining
// pixels will alternate between black and white.
// Note that both width and height must be multiples of 8
std::vector<Rgba> makeCheckerboard(int width, int height)
{
    ASSERT(width % 8 == 0 && height % 8 == 0);

    const Rgba white    = {0xFF, 0xFF, 0xFF, 0xFF};
    const Rgba black    = {0, 0, 0, 0xFF};
    const Rgba colors[] = {white, black};

    std::vector<Rgba> result;
    result.reserve(width * height);

    int colorIndex = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            result.push_back(colors[colorIndex]);
            colorIndex ^= 1UL;  // toggle the last bit, so we alternate between 0 and 1;
        }
        colorIndex ^= 1UL;
    }
    return result;
}

// Similar to makeCheckerboard(), but returns an ASTC-encoded image instead, with 8x8 block size.
std::vector<uint8_t> makeAstcCheckerboard(int width, int height)
{
    ASSERT(width % 8 == 0 && height % 8 == 0);

    // One 8x8 ASTC block with a checkerboard pattern (alternating black and white pixels)
    const std::vector<uint8_t> block = {0x44, 0x05, 0x00, 0xfe, 0x01, 0x00, 0x00, 0x00,
                                        0x55, 0xaa, 0x55, 0xaa, 0x55, 0xaa, 0x55, 0xaa};

    const int numBlocks = width * height / (8 * 8);

    std::vector<uint8_t> result;
    result.reserve(numBlocks * block.size());
    for (int i = 0; i < numBlocks; ++i)
    {
        result.insert(result.end(), block.begin(), block.end());
    }

    return result;
}

}  // namespace testing