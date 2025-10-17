/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#include "ImageBufferSkiaUnacceleratedBackend.h"

#if USE(SKIA)
#include "FontRenderOptions.h"
#include "IntRect.h"
#include "PixelBuffer.h"
#include "SkiaSpanExtras.h"
#include <skia/core/SkPixmap.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBufferSkiaUnacceleratedBackend);

std::unique_ptr<ImageBufferSkiaUnacceleratedBackend> ImageBufferSkiaUnacceleratedBackend::create(const Parameters& parameters, const ImageBufferCreationContext&)
{
    IntSize backendSize = calculateSafeBackendSize(parameters);
    if (backendSize.isEmpty())
        return nullptr;

    auto imageInfo = SkImageInfo::MakeN32Premul(backendSize.width(), backendSize.height(), parameters.colorSpace.platformColorSpace());
    SkSurfaceProps properties = { 0, FontRenderOptions::singleton().subpixelOrder() };
    auto surface = SkSurfaces::Raster(imageInfo, &properties);
    if (!surface || !surface->getCanvas())
        return nullptr;

    return std::unique_ptr<ImageBufferSkiaUnacceleratedBackend>(new ImageBufferSkiaUnacceleratedBackend(parameters, WTFMove(surface)));
}

ImageBufferSkiaUnacceleratedBackend::ImageBufferSkiaUnacceleratedBackend(const Parameters& parameters, sk_sp<SkSurface>&& surface)
    : ImageBufferSkiaSurfaceBackend(parameters, WTFMove(surface), RenderingMode::Unaccelerated)
{
}

ImageBufferSkiaUnacceleratedBackend::~ImageBufferSkiaUnacceleratedBackend() = default;

RefPtr<NativeImage> ImageBufferSkiaUnacceleratedBackend::copyNativeImage()
{
    SkPixmap pixmap;
    if (m_surface->peekPixels(&pixmap))
        return NativeImage::create(SkImages::RasterFromPixmapCopy(pixmap));
    return nullptr;
}

RefPtr<NativeImage> ImageBufferSkiaUnacceleratedBackend::createNativeImageReference()
{
    SkPixmap pixmap;
    if (m_surface->peekPixels(&pixmap)) {
        return NativeImage::create(SkImages::RasterFromPixmap(pixmap, [](const void*, void* context) {
            static_cast<SkSurface*>(context)->unref();
        }, SkSafeRef(m_surface.get())));
    }
    return nullptr;
}

void ImageBufferSkiaUnacceleratedBackend::getPixelBuffer(const IntRect& srcRect, PixelBuffer& destination)
{
    SkPixmap pixmap;
    if (m_surface->peekPixels(&pixmap))
        ImageBufferBackend::getPixelBuffer(srcRect, span(pixmap), destination);
}

void ImageBufferSkiaUnacceleratedBackend::putPixelBuffer(const PixelBuffer& pixelBuffer, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat)
{
    SkPixmap pixmap;
    if (m_surface->peekPixels(&pixmap))
        ImageBufferBackend::putPixelBuffer(pixelBuffer, srcRect, destPoint, destFormat, mutableSpan(pixmap));
}

} // namespace WebCore

#endif // USE(SKIA)
