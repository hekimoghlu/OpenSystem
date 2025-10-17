/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include "ShareableBitmapUtilities.h"

#include <WebCore/CachedImage.h>
#include <WebCore/FrameSnapshotting.h>
#include <WebCore/GeometryUtilities.h>
#include <WebCore/GraphicsContext.h>
#include <WebCore/ImageBuffer.h>
#include <WebCore/IntSize.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/PlatformScreen.h>
#include <WebCore/RenderElementInlines.h>
#include <WebCore/RenderImage.h>
#include <WebCore/RenderVideo.h>
#include <WebCore/ShareableBitmap.h>

namespace WebKit {
using namespace WebCore;

RefPtr<ShareableBitmap> createShareableBitmap(RenderImage& renderImage, CreateShareableBitmapFromImageOptions&& options)
{
    Ref frame = renderImage.frame();
    auto colorSpaceForBitmap = screenColorSpace(frame->mainFrame().virtualView());
    if (!renderImage.isRenderMedia() && !renderImage.opacity() && options.useSnapshotForTransparentImages == UseSnapshotForTransparentImages::Yes) {
        auto snapshotRect = renderImage.absoluteBoundingBoxRect();
        if (snapshotRect.isEmpty())
            return { };

        OptionSet<SnapshotFlags> snapshotFlags { SnapshotFlags::ExcludeSelectionHighlighting, SnapshotFlags::PaintEverythingExcludingSelection };
        auto imageBuffer = snapshotFrameRect(frame.get(), snapshotRect, { snapshotFlags, ImageBufferPixelFormat::BGRA8, DestinationColorSpace::SRGB() });
        if (!imageBuffer)
            return { };

        auto snapshotImage = ImageBuffer::sinkIntoNativeImage(WTFMove(imageBuffer));
        if (!snapshotImage)
            return { };

        auto bitmap = ShareableBitmap::create({ snapshotImage->size(), WTFMove(colorSpaceForBitmap) });
        if (!bitmap)
            return { };

        auto context = bitmap->createGraphicsContext();
        if (!context)
            return { };
        FloatRect imageRect { { }, snapshotImage->size() };
        context->drawNativeImage(*snapshotImage, imageRect, imageRect);
        return bitmap;
    }

#if ENABLE(VIDEO)
    if (auto* renderVideo = dynamicDowncast<RenderVideo>(renderImage)) {
        Ref video = renderVideo->videoElement();
        auto image = video->nativeImageForCurrentTime();
        if (!image)
            return { };

        auto imageSize = image->size();
        if (imageSize.isEmpty() || imageSize.width() <= 1 || imageSize.height() <= 1)
            return { };

        auto bitmap = ShareableBitmap::create({ imageSize, WTFMove(colorSpaceForBitmap) });
        if (!bitmap)
            return { };

        auto context = bitmap->createGraphicsContext();
        if (!context)
            return { };

        context->drawNativeImage(*image, FloatRect { { }, imageSize }, FloatRect { { }, imageSize });
        return bitmap;
    }
#endif // ENABLE(VIDEO)

    auto* cachedImage = renderImage.cachedImage();
    if (!cachedImage || cachedImage->errorOccurred())
        return { };

    auto* image = cachedImage->imageForRenderer(&renderImage);
    if (!image || image->width() <= 1 || image->height() <= 1)
        return { };

    if (options.allowAnimatedImages == AllowAnimatedImages::No && image->isAnimated())
        return { };

    auto bitmapSize = cachedImage->imageSizeForRenderer(&renderImage);
    if (options.screenSizeInPixels) {
        auto scaledSize = largestRectWithAspectRatioInsideRect(bitmapSize.width() / bitmapSize.height(), { FloatPoint(), *options.screenSizeInPixels }).size();
        bitmapSize = scaledSize.width() < bitmapSize.width() ? scaledSize : bitmapSize;
    }

    // FIXME: Only select ExtendedColor on images known to need wide gamut.
    auto sharedBitmap = ShareableBitmap::create({ IntSize(bitmapSize), WTFMove(colorSpaceForBitmap) });
    if (!sharedBitmap)
        return { };

    auto graphicsContext = sharedBitmap->createGraphicsContext();
    if (!graphicsContext)
        return { };

    graphicsContext->drawImage(*image, FloatRect(0, 0, bitmapSize.width(), bitmapSize.height()), { renderImage.imageOrientation() });
    return sharedBitmap;
}

}
