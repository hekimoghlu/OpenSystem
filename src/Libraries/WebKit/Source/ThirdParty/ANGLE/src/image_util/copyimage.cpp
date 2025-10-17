/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// copyimage.cpp: Defines image copying functions

#include "image_util/copyimage.h"

namespace angle
{

namespace
{
inline uint32_t SwizzleBGRAToRGBA(uint32_t argb)
{
    return ((argb & 0x000000FF) << 16) |  // Move BGRA blue to RGBA blue
           ((argb & 0x00FF0000) >> 16) |  // Move BGRA red to RGBA red
           ((argb & 0xFF00FF00));         // Keep alpha and green
}

void CopyBGRA8ToRGBA8Fast(const uint8_t *source,
                          int srcYAxisPitch,
                          uint8_t *dest,
                          int destYAxisPitch,
                          int destWidth,
                          int destHeight)
{
    for (int y = 0; y < destHeight; ++y)
    {
        const uint32_t *src32 = reinterpret_cast<const uint32_t *>(source + y * srcYAxisPitch);
        uint32_t *dest32      = reinterpret_cast<uint32_t *>(dest + y * destYAxisPitch);
        const uint32_t *end32 = src32 + destWidth;
        while (src32 != end32)
        {
            *dest32++ = SwizzleBGRAToRGBA(*src32++);
        }
    }
}
}  // namespace

void CopyBGRA8ToRGBA8(const uint8_t *source,
                      int srcXAxisPitch,
                      int srcYAxisPitch,
                      uint8_t *dest,
                      int destXAxisPitch,
                      int destYAxisPitch,
                      int destWidth,
                      int destHeight)
{
    if (srcXAxisPitch == 4 && destXAxisPitch == 4)
    {
        CopyBGRA8ToRGBA8Fast(source, srcYAxisPitch, dest, destYAxisPitch, destWidth, destHeight);
        return;
    }

    for (int y = 0; y < destHeight; ++y)
    {
        uint8_t *dst       = dest + y * destYAxisPitch;
        const uint8_t *src = source + y * srcYAxisPitch;
        const uint8_t *end = src + destWidth * srcXAxisPitch;

        while (src != end)
        {
            *reinterpret_cast<uint32_t *>(dst) =
                SwizzleBGRAToRGBA(*reinterpret_cast<const uint32_t *>(src));
            src += srcXAxisPitch;
            dst += destXAxisPitch;
        }
    }
}

}  // namespace angle
