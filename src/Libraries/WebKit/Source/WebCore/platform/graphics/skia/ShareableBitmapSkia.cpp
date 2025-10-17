/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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
#include "FontRenderOptions.h"
#include "GraphicsContextSkia.h"
#include "NotImplemented.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN // GLib/Win ports
#include <skia/core/SkSurface.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

std::optional<DestinationColorSpace> ShareableBitmapConfiguration::validateColorSpace(std::optional<DestinationColorSpace> colorSpace)
{
    return colorSpace;
}

CheckedUint32 ShareableBitmapConfiguration::calculateBytesPerPixel(const DestinationColorSpace& colorSpace)
{
    return SkImageInfo::MakeN32Premul(1, 1, colorSpace.platformColorSpace()).bytesPerPixel();
}

CheckedUint32 ShareableBitmapConfiguration::calculateBytesPerRow(const IntSize& size, const DestinationColorSpace& colorSpace)
{
    return SkImageInfo::MakeN32Premul(size.width(), size.height(), colorSpace.platformColorSpace()).minRowBytes();
}

std::unique_ptr<GraphicsContext> ShareableBitmap::createGraphicsContext()
{
    ref();
    SkSurfaceProps properties = { 0, FontRenderOptions::singleton().subpixelOrder() };
    auto surface = SkSurfaces::WrapPixels(m_configuration.imageInfo(), mutableSpan().data(), bytesPerRow(), [](void*, void* context) {
        static_cast<ShareableBitmap*>(context)->deref();
    }, this, &properties);

    auto* canvas = surface->getCanvas();
    if (!canvas)
        return nullptr;

    return makeUnique<GraphicsContextSkia>(*canvas, RenderingMode::Unaccelerated, RenderingPurpose::ShareableSnapshot, [surface = WTFMove(surface)] { });
}

void ShareableBitmap::paint(GraphicsContext& context, const IntPoint& dstPoint, const IntRect& srcRect)
{
    paint(context, 1, dstPoint, srcRect);
}

void ShareableBitmap::paint(GraphicsContext& context, float scaleFactor, const IntPoint& dstPoint, const IntRect& srcRect)
{
    FloatRect scaledSrcRect(srcRect);
    scaledSrcRect.scale(scaleFactor);
    FloatRect scaledDestRect(dstPoint, srcRect.size());
    scaledDestRect.scale(scaleFactor);
    auto image = createPlatformImage(BackingStoreCopy::DontCopyBackingStore);
    context.platformContext()->drawImageRect(image.get(), scaledSrcRect, scaledDestRect, { }, nullptr, { });
}

RefPtr<Image> ShareableBitmap::createImage()
{
    return BitmapImage::create(createPlatformImage(BackingStoreCopy::DontCopyBackingStore));
}

PlatformImagePtr ShareableBitmap::createPlatformImage(BackingStoreCopy backingStoreCopy, ShouldInterpolate)
{
    sk_sp<SkData> pixelData;
    if (backingStoreCopy == BackingStoreCopy::CopyBackingStore)
        pixelData = SkData::MakeWithCopy(span().data(), sizeInBytes());
    else {
        ref();
        pixelData = SkData::MakeWithProc(span().data(), sizeInBytes(), [](const void*, void* bitmap) -> void {
            static_cast<ShareableBitmap*>(bitmap)->deref();
        }, this);
    }
    return SkImages::RasterFromData(m_configuration.imageInfo(), pixelData, bytesPerRow());
}

void ShareableBitmap::setOwnershipOfMemory(const ProcessIdentity&)
{
}

} // namespace WebCore
