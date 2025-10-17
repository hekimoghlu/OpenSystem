/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#include "PixelBufferFormat.h"
#include <span>

namespace WebCore {

class IntSize;

struct PixelBufferConversionView {
    PixelBufferFormat format;
    unsigned bytesPerRow;
    std::span<uint8_t> rows;
};

struct ConstPixelBufferConversionView {
    PixelBufferFormat format;
    unsigned bytesPerRow;
    std::span<const uint8_t> rows;
};

WEBCORE_EXPORT void convertImagePixels(const ConstPixelBufferConversionView& source, const PixelBufferConversionView& destination, const IntSize&);

WEBCORE_EXPORT void copyRowsInternal(unsigned sourceBytesPerRow, std::span<const uint8_t> source, unsigned destinationBytesPerRow, std::span<uint8_t> destination, unsigned rows, unsigned copyBytesPerRow);

inline void copyRows(unsigned sourceBytesPerRow, std::span<const uint8_t> source, unsigned destinationBytesPerRow, std::span<uint8_t> destination, unsigned rows, unsigned copyBytesPerRow)
{
    if (!rows || !copyBytesPerRow)
        return;
    unsigned requiredSourceBytes = sourceBytesPerRow * (rows - 1) + copyBytesPerRow;
    if (source.size() < requiredSourceBytes) {
        ASSERT_NOT_REACHED();
        return;
    }
    unsigned requiredDestinationBytes = destinationBytesPerRow * (rows - 1) + copyBytesPerRow;
    if (destination.size() < requiredDestinationBytes) {
        ASSERT_NOT_REACHED();
        return;
    }
    if (rows > 1 && (sourceBytesPerRow < copyBytesPerRow || destinationBytesPerRow < copyBytesPerRow)) {
        ASSERT_NOT_REACHED();
        return;
    }
    copyRowsInternal(sourceBytesPerRow, source, destinationBytesPerRow, destination, rows, copyBytesPerRow);
}

}
