/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#include "ImageBufferCairoSurfaceBackend.h"

#include "BitmapImage.h"
#include "CairoOperations.h"
#include "CairoUtilities.h"
#include "Color.h"
#include "GraphicsContext.h"
#include "ImageBufferUtilitiesCairo.h"
#include "PixelBuffer.h"
#include <cairo.h>

#if USE(CAIRO)

namespace WebCore {

ImageBufferCairoSurfaceBackend::ImageBufferCairoSurfaceBackend(const Parameters& parameters, RefPtr<cairo_surface_t>&& surface)
    : ImageBufferCairoBackend(parameters)
    , m_surface(WTFMove(surface))
    , m_context(m_surface.get())
{
    ASSERT(cairo_surface_status(m_surface.get()) == CAIRO_STATUS_SUCCESS);
    m_context.applyDeviceScaleFactor(parameters.resolutionScale);
}

GraphicsContext& ImageBufferCairoSurfaceBackend::context()
{
    return m_context;
}

unsigned ImageBufferCairoSurfaceBackend::bytesPerRow() const
{
    return cairo_image_surface_get_stride(m_surface.get());
}

RefPtr<NativeImage> ImageBufferCairoSurfaceBackend::copyNativeImage()
{
    auto copy = adoptRef(cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
    cairo_image_surface_get_width(m_surface.get()),
    cairo_image_surface_get_height(m_surface.get())));

    auto cr = adoptRef(cairo_create(copy.get()));
    cairo_set_operator(cr.get(), CAIRO_OPERATOR_SOURCE);
    cairo_set_source_surface(cr.get(), m_surface.get(), 0, 0);
    cairo_paint(cr.get());

    return NativeImage::create(WTFMove(copy));
}

RefPtr<NativeImage> ImageBufferCairoSurfaceBackend::createNativeImageReference()
{
    return NativeImage::create(RefPtr { m_surface.get() });
}

bool ImageBufferCairoSurfaceBackend::canMapBackingStore() const
{
    return true;
}

RefPtr<cairo_surface_t> ImageBufferCairoSurfaceBackend::createCairoSurface()
{
    return RefPtr { m_surface.get() };
}

RefPtr<NativeImage> ImageBufferCairoSurfaceBackend::cairoSurfaceCoerceToImage()
{
    if (cairo_surface_get_type(m_surface.get()) == CAIRO_SURFACE_TYPE_IMAGE && cairo_surface_get_content(m_surface.get()) == CAIRO_CONTENT_COLOR_ALPHA)
        return createNativeImageReference();
    return copyNativeImage();
}

void ImageBufferCairoSurfaceBackend::getPixelBuffer(const IntRect& srcRect, PixelBuffer& destination)
{
    ImageBufferBackend::getPixelBuffer(srcRect, span(m_surface.get()), destination);
}

void ImageBufferCairoSurfaceBackend::putPixelBuffer(const PixelBuffer& pixelBuffer, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat)
{
    ImageBufferBackend::putPixelBuffer(pixelBuffer, srcRect, destPoint, destFormat, mutableSpan(m_surface.get()));

    cairo_surface_mark_dirty_rectangle(m_surface.get(), destPoint.x(), destPoint.y(), srcRect.width(), srcRect.height());
}

String ImageBufferCairoSurfaceBackend::debugDescription() const
{
    TextStream stream;
    stream << "ImageBufferCairoSurfaceBackend " << this << " " << m_surface.get();
    return stream.release();
}

} // namespace WebCore

#endif // USE(CAIRO)
