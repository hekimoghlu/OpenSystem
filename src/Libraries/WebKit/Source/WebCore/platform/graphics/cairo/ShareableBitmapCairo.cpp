/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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
#include "ShareableBitmap.h"

#include "BitmapImage.h"
#include "CairoOperations.h"
#include "CairoUtilities.h"
#include "GraphicsContextCairo.h"
#include "NotImplemented.h"

namespace WebCore {

static const cairo_format_t cairoFormat = CAIRO_FORMAT_ARGB32;

std::optional<DestinationColorSpace> ShareableBitmapConfiguration::validateColorSpace(std::optional<DestinationColorSpace> colorSpace)
{
    return colorSpace;
}

CheckedUint32 ShareableBitmapConfiguration::calculateBytesPerPixel(const DestinationColorSpace&)
{
    return 4;
}

CheckedUint32 ShareableBitmapConfiguration::calculateBytesPerRow(const IntSize& size, const DestinationColorSpace&)
{
    return cairo_format_stride_for_width(cairoFormat, size.width());
}

static inline RefPtr<cairo_surface_t> createSurfaceFromData(uint8_t* data, const IntSize& size)
{
    const int stride = cairo_format_stride_for_width(cairoFormat, size.width());
    return adoptRef(cairo_image_surface_create_for_data(data, cairoFormat, size.width(), size.height(), stride));
}

std::unique_ptr<GraphicsContext> ShareableBitmap::createGraphicsContext()
{
    RefPtr<cairo_surface_t> image = createCairoSurface();
    return makeUnique<GraphicsContextCairo>(image.get());
}

void ShareableBitmap::paint(GraphicsContext& context, const IntPoint& dstPoint, const IntRect& srcRect)
{
    paint(context, 1, dstPoint, srcRect);
}

void ShareableBitmap::paint(GraphicsContext& context, float scaleFactor, const IntPoint& dstPoint, const IntRect& srcRect)
{
    RefPtr<cairo_surface_t> surface = createSurfaceFromData(mutableSpan().data(), size());
    cairo_surface_set_device_scale(surface.get(), scaleFactor, scaleFactor);
    FloatRect destRect(dstPoint, srcRect.size());

    ASSERT(context.hasPlatformContext());
    auto& state = context.state();
    Cairo::drawSurface(*context.platformContext(), surface.get(), destRect, srcRect, state.imageInterpolationQuality(), state.alpha(), Cairo::ShadowState(state));
}

RefPtr<cairo_surface_t> ShareableBitmap::createPersistentCairoSurface()
{
    return createSurfaceFromData(mutableSpan().data(), size());
}

RefPtr<cairo_surface_t> ShareableBitmap::createCairoSurface()
{
    RefPtr<cairo_surface_t> image = createSurfaceFromData(mutableSpan().data(), size());

    ref(); // Balanced by deref in releaseSurfaceData.
    static cairo_user_data_key_t dataKey;
    cairo_surface_set_user_data(image.get(), &dataKey, this, releaseSurfaceData);
    return image;
}

void ShareableBitmap::releaseSurfaceData(void* typelessBitmap)
{
    static_cast<ShareableBitmap*>(typelessBitmap)->deref(); // Balanced by ref in createCairoSurface.
}

RefPtr<Image> ShareableBitmap::createImage()
{
    RefPtr<cairo_surface_t> surface = createCairoSurface();
    if (!surface)
        return nullptr;

    return BitmapImage::create(WTFMove(surface));
}

void ShareableBitmap::setOwnershipOfMemory(const ProcessIdentity&)
{
}

} // namespace WebCore
