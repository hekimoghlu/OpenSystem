/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#include "ImageAdapter.h"

#if PLATFORM(WIN) && USE(CAIRO)

#include "GraphicsContextCairo.h"
#include <cairo-win32.h>

namespace WebCore {

RefPtr<NativeImage> ImageAdapter::nativeImageOfHBITMAP(HBITMAP bmp)
{
    DIBSECTION dibSection;
    if (!GetObject(bmp, sizeof(DIBSECTION), &dibSection))
        return nullptr;

    ASSERT(dibSection.dsBm.bmBitsPixel == 32);
    if (dibSection.dsBm.bmBitsPixel != 32)
        return nullptr;

    ASSERT(dibSection.dsBm.bmBits);
    if (!dibSection.dsBm.bmBits)
        return nullptr;

    auto surface = adoptRef(cairo_win32_surface_create_with_dib(CAIRO_FORMAT_ARGB32, dibSection.dsBm.bmWidth, dibSection.dsBm.bmHeight));
    return NativeImage::create(WTFMove(surface));
}

bool ImageAdapter::getHBITMAPOfSize(HBITMAP bmp, const IntSize* size)
{
    ASSERT(bmp);

    BITMAP bmpInfo;
    GetObject(bmp, sizeof(BITMAP), &bmpInfo);

    // If this is a 32bpp bitmap, which it always should be, we'll clear it so alpha-wise it will be visible
    if (bmpInfo.bmBitsPixel == 32 && bmpInfo.bmBits) {
        int bufferSize = bmpInfo.bmWidthBytes * bmpInfo.bmHeight;
        memset(bmpInfo.bmBits, 255, bufferSize);
    }

    unsigned char* bmpdata = (unsigned char*)bmpInfo.bmBits + bmpInfo.bmWidthBytes * (bmpInfo.bmHeight - 1);
    auto platformImage = adoptRef(cairo_image_surface_create_for_data(bmpdata, CAIRO_FORMAT_ARGB32, bmpInfo.bmWidth, bmpInfo.bmHeight, -bmpInfo.bmWidthBytes));

    GraphicsContextCairo gc(platformImage.get());

    auto imageSize = image().size();
    auto destinationRect = FloatRect(0.0f, 0.0f, bmpInfo.bmWidth, bmpInfo.bmHeight);

    if (auto nativeImage = size ? nativeImageOfSize(*size) : nullptr) {
        auto sourceRect = FloatRect { { }, *size };
        gc.drawNativeImage(*nativeImage, destinationRect, sourceRect, { CompositeOperator::Copy });
        return true;
    }

    auto sourceRect = FloatRect { { }, imageSize };
    gc.drawImage(image(), destinationRect, sourceRect, { CompositeOperator::Copy });
    return true;
}

} // namespace WebCore

#endif // PLATFORM(WIN) && USE(CAIRO)
