/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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
#include "DragImage.h"

#include "BitmapInfo.h"
#include "CachedImage.h"
#include "GraphicsContextCairo.h"
#include "HWndDC.h"
#include "Image.h"
#include <cairo-win32.h>
#include <windows.h>
#include <wtf/RetainPtr.h>
#include <wtf/win/GDIObject.h>

namespace WebCore {

void deallocContext(GraphicsContextCairo* target)
{
    delete target;
}

GDIObject<HBITMAP> allocImage(HDC dc, IntSize size, GraphicsContextCairo** targetRef)
{
    BitmapInfo bmpInfo = BitmapInfo::create(size);

    LPVOID bits;
    auto hbmp = adoptGDIObject(::CreateDIBSection(dc, &bmpInfo, DIB_RGB_COLORS, &bits, 0, 0));

    // At this point, we have a Cairo surface that points to a Windows DIB. The DIB interprets
    // with the opposite meaning of positive Y axis, so everything we draw into this cairo
    // context is going to be upside down.
    if (!targetRef)
        return hbmp;

    cairo_surface_t* bitmapContext = cairo_image_surface_create_for_data((unsigned char*)bits,
        CAIRO_FORMAT_ARGB32,
        bmpInfo.bmiHeader.biWidth,
        bmpInfo.bmiHeader.biHeight,
        bmpInfo.bmiHeader.biWidth * 4);

    if (!bitmapContext)
        return GDIObject<HBITMAP>();

    cairo_t* cr = cairo_create(bitmapContext);
    cairo_surface_destroy(bitmapContext);

    // At this point, we have a Cairo surface that points to a Windows DIB. The DIB interprets
    // with the opposite meaning of positive Y axis, so everything we draw into this cairo
    // context is going to be upside down.
    //
    // So, we must invert the CTM for the context so that drawing commands will be flipped
    // before they get written to the internal buffer.
    cairo_matrix_t matrix;
    cairo_matrix_init(&matrix, 1.0, 0.0, 0.0, -1.0, 0.0, size.height());
    cairo_set_matrix(cr, &matrix);

    *targetRef = new PlatformGraphicsContext(cr);
    cairo_destroy(cr);

    return hbmp;
}

static cairo_surface_t* createCairoContextFromBitmap(HBITMAP bitmap)
{
    BITMAP info;
    GetObject(bitmap, sizeof(info), &info);
    ASSERT(info.bmBitsPixel == 32);

    // At this point, we have a Cairo surface that points to a Windows BITMAP. The BITMAP
    // has the opposite meaning of positive Y axis, so everything we draw into this cairo
    // context is going to be upside down.
    return cairo_image_surface_create_for_data((unsigned char*)info.bmBits,
        CAIRO_FORMAT_ARGB32,
        info.bmWidth,
        info.bmHeight,
        info.bmWidthBytes);
}

DragImageRef scaleDragImage(DragImageRef imageRef, FloatSize scale)
{
    // FIXME: due to the way drag images are done on windows we need
    // to preprocess the alpha channel <rdar://problem/5015946>
    if (!imageRef)
        return 0;

    auto image = adoptGDIObject(imageRef);

    IntSize srcSize = dragImageSize(image.get());
    IntSize dstSize(static_cast<int>(srcSize.width() * scale.width()), static_cast<int>(srcSize.height() * scale.height()));

    HWndDC dc(0);
    auto dstDC = adoptGDIObject(::CreateCompatibleDC(dc));
    if (!dstDC)
        return image.leak();

    GraphicsContextCairo* targetContext;
    GDIObject<HBITMAP> hbmp = allocImage(dstDC.get(), dstSize, &targetContext);
    if (!hbmp)
        return image.leak();

    cairo_surface_t* srcImage = createCairoContextFromBitmap(image.get());

    // Scale the target surface to the new image size, and flip it
    // so that when we set the srcImage as the surface it will draw
    // right-side-up.
    cairo_t* cr = targetContext->cr();
    cairo_translate(cr, 0, dstSize.height());
    cairo_scale(cr, scale.width(), -scale.height());
    cairo_set_source_surface(cr, srcImage, 0.0, 0.0);

    // Now we can paint and get the correct result
    cairo_paint(cr);

    cairo_surface_destroy(srcImage);
    deallocContext(targetContext);

    return hbmp.leak();
}

DragImageRef createDragImageFromImage(Image* img, ImageOrientation)
{
    HWndDC dc(0);
    auto workingDC = adoptGDIObject(::CreateCompatibleDC(dc));
    if (!workingDC)
        return 0;

    GraphicsContextCairo* drawContext = 0;
    auto hbmp = allocImage(workingDC.get(), IntSize(img->size()), &drawContext);
    if (!hbmp || !drawContext)
        return 0;

    cairo_t* cr = drawContext->cr();
    cairo_set_source_rgb(cr, 1.0, 0.0, 1.0);
    cairo_fill_preserve(cr);

    if (auto nativeImage = img->currentNativeImage()) {
        auto& surface = nativeImage->platformImage();
        // Draw the image.
        cairo_set_source_surface(cr, surface.get(), 0.0, 0.0);
        cairo_paint(cr);
    }

    deallocContext(drawContext);

    return hbmp.leak();
}

}
