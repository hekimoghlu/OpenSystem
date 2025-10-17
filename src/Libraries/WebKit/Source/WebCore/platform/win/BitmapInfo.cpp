/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#include "config.h"
#include "BitmapInfo.h"

#include <wtf/Assertions.h>

namespace WebCore {

BitmapInfo bitmapInfoForSize(int width, int height, BitmapInfo::BitCount bitCount)
{
    BitmapInfo bitmapInfo;
    bitmapInfo.bmiHeader.biWidth         = width;
    bitmapInfo.bmiHeader.biHeight        = height;
    bitmapInfo.bmiHeader.biPlanes        = 1;
    bitmapInfo.bmiHeader.biBitCount      = bitCount;
    bitmapInfo.bmiHeader.biCompression   = BI_RGB;

    return bitmapInfo;
}

BitmapInfo::BitmapInfo()
{
    memset(&bmiHeader, 0, sizeof(bmiHeader));
    bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
}

BitmapInfo BitmapInfo::create(const IntSize& size, BitCount bitCount)
{
    return bitmapInfoForSize(size.width(), size.height(), bitCount);
}

BitmapInfo BitmapInfo::createBottomUp(const IntSize& size, BitCount bitCount)
{
    return bitmapInfoForSize(size.width(), -size.height(), bitCount);
}

} // namespace WebCore
