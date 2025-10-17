/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#ifndef BitmapInfo_h
#define BitmapInfo_h

#include "IntSize.h"
#include <windows.h>

namespace WebCore {

struct BitmapInfo : public BITMAPINFO {
    enum BitCount {
        BitCount1 = 1,
        BitCount4 = 4,
        BitCount8 = 8,
        BitCount16 = 16,
        BitCount24 = 24,
        BitCount32 = 32
    };

    BitmapInfo();
    WEBCORE_EXPORT static BitmapInfo create(const IntSize&, BitCount bitCount = BitCount32);
    WEBCORE_EXPORT static BitmapInfo createBottomUp(const IntSize&, BitCount bitCount = BitCount32);

    bool is16bit() const { return bmiHeader.biBitCount == 16; }
    bool is32bit() const { return bmiHeader.biBitCount == 32; }
    unsigned width() const { return std::abs(bmiHeader.biWidth); }
    unsigned height() const { return std::abs(bmiHeader.biHeight); }
    IntSize size() const { return IntSize(width(), height()); }
    unsigned bytesPerLine() const { return (width() * bmiHeader.biBitCount + 7) / 8; }
    unsigned paddedBytesPerLine() const { return (bytesPerLine() + 3) & ~0x3; }
    unsigned paddedWidth() const { return paddedBytesPerLine() * 8 / bmiHeader.biBitCount; }
    unsigned numPixels() const { return paddedWidth() * height(); }
};

} // namespace WebCore

#endif // BitmapInfo_h
