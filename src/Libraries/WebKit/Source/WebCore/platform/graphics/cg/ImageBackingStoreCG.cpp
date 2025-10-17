/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
#include "ImageBackingStore.h"

namespace WebCore {

static void dataProviderReleaseCallback(void* info, const void*, size_t)
{
    auto* pixels = static_cast<FragmentedSharedBuffer::DataSegment*>(info);
    pixels->deref(); // Balanced below in ImageBackingStore::image().
}

PlatformImagePtr ImageBackingStore::image() const
{
    static const size_t bytesPerPixel = 4;
    static const size_t bitsPerComponent = 8;
    size_t width = size().width();
    size_t height = size().height();
    size_t bytesPerRow = bytesPerPixel * width;

    auto colorSpace = adoptCF(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    auto dataProvider = adoptCF(CGDataProviderCreateWithData(m_pixels.get(), m_pixelsSpan.data(), height * bytesPerRow, dataProviderReleaseCallback));

    if (!dataProvider)
        return nullptr;

    m_pixels->ref(); // Balanced above in dataProviderReleaseCallback().
IGNORE_WARNINGS_BEGIN("deprecated-enum-enum-conversion")
    CGBitmapInfo bitmapInfo = (m_premultiplyAlpha ? kCGImageAlphaPremultipliedFirst : kCGImageAlphaFirst) | kCGImageByteOrder32Little;
IGNORE_WARNINGS_END
    return adoptCF(CGImageCreate(width, height, bitsPerComponent, bytesPerPixel * 8, bytesPerRow, colorSpace.get(), bitmapInfo, dataProvider.get(), nullptr, true, kCGRenderingIntentDefault));
}

} // namespace WebCore
