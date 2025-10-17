/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#pragma once

#include "IntRect.h"
#include "IntSize.h"
#include <windows.h>

typedef unsigned char UInt8;

namespace WebCore {

class DIBPixelData {
public:
    DIBPixelData()
        : m_bitmapBuffer(0)
        , m_bitmapBufferLength(0)
        , m_bytesPerRow(0)
        , m_bitsPerPixel(0)
    {
    }
    DIBPixelData(HBITMAP);
    DIBPixelData(void* data, IntSize);

    void initialize(HBITMAP);

#ifndef NDEBUG
    void writeToFile(LPCWSTR);
#endif

    UInt8* buffer() const { return m_bitmapBuffer; }
    unsigned bufferLength() const { return m_bitmapBufferLength; }
    const IntSize& size() const { return m_size; }
    unsigned bytesPerRow() const { return m_bytesPerRow; }
    unsigned short bitsPerPixel() const { return m_bitsPerPixel; }
    WEBCORE_EXPORT static void setRGBABitmapAlpha(HDC, const IntRect&, unsigned char);

private:
    UInt8* m_bitmapBuffer;
    unsigned m_bitmapBufferLength;
    IntSize m_size;
    unsigned m_bytesPerRow;
    unsigned short m_bitsPerPixel;
};

} // namespace WebCore
