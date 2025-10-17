/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#include "BitmapImage.h"

#include "BitmapImageSource.h"
#include "GeometryUtilities.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "ImageObserver.h"
#include "NativeImageSource.h"

namespace WebCore {

Ref<BitmapImage> BitmapImage::create(ImageObserver* observer, AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
{
    return adoptRef(*new BitmapImage(observer, alphaOption, gammaAndColorProfileOption));
}

Ref<BitmapImage> BitmapImage::create(Ref<NativeImage>&& nativeImage)
{
    return adoptRef(*new BitmapImage(WTFMove(nativeImage)));
}

RefPtr<BitmapImage> BitmapImage::create(RefPtr<NativeImage>&& nativeImage)
{
    if (!nativeImage)
        return nullptr;
    return adoptRef(*new BitmapImage(nativeImage.releaseNonNull()));
}

RefPtr<BitmapImage> BitmapImage::create(PlatformImagePtr&& platformImage)
{
    return create(NativeImage::create(WTFMove(platformImage)));
}

BitmapImage::BitmapImage(ImageObserver* observer, AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    : Image(observer)
    , m_source(BitmapImageSource::create(*this, alphaOption, gammaAndColorProfileOption))
{
}

BitmapImage::BitmapImage(Ref<NativeImage>&& image)
    : m_source(NativeImageSource::create(WTFMove(image)))
{
}

EncodedDataStatus BitmapImage::dataChanged(bool allDataReceived)
{
    return m_source->dataChanged(data(), allDataReceived);
}

void BitmapImage::destroyDecodedData(bool destroyAll)
{
    m_source->destroyDecodedData(destroyAll);
    invalidateAdapter();
}

ImageDrawResult BitmapImage::draw(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    if (destinationRect.isEmpty() || sourceRect.isEmpty())
        return ImageDrawResult::DidNothing;

    auto size = m_source->size(ImageOrientation::Orientation::None);
    auto sourceSize = m_source->sourceSize(ImageOrientation::Orientation::None);

    // adjustedSourceRect is in the coordinates of densityCorrectedSize, so map it to the sourceSize.
    auto adjustedSourceRect = sourceRect;
    if (sourceSize != size)
        adjustedSourceRect.scale(sourceSize / size);

    auto scaleFactorForDrawing = context.scaleFactorForDrawing(destinationRect, adjustedSourceRect);
    auto sizeForDrawing = expandedIntSize(sourceSize * scaleFactorForDrawing);
    auto subsamplingLevel =  m_source->subsamplingLevelForScaleFactor(context, scaleFactorForDrawing, options.allowImageSubsampling());

    auto nativeImage = m_source->currentNativeImageForDrawing(subsamplingLevel, { options.decodingMode(), sizeForDrawing });

    if (!nativeImage) {
        if (nativeImage.error() != DecodingStatus::Decoding)
            return ImageDrawResult::DidNothing;

        if (options.showDebugBackground() == ShowDebugBackground::Yes)
            fillWithSolidColor(context, destinationRect, Color::yellow.colorWithAlphaByte(128), options.compositeOperator());

        return ImageDrawResult::DidRequestDecoding;
    }

    if (auto color = (*nativeImage)->singlePixelSolidColor())
        fillWithSolidColor(context, destinationRect, *color, options.compositeOperator());
    else {
        // adjustedSourceRect is in the coordinates of the unsubsampled image, so map it to the subsampled image.
        auto imageSize = (*nativeImage)->size();
        if (imageSize != sourceSize)
            adjustedSourceRect.scale(imageSize / sourceSize);

        auto orientation = options.orientation();
        if (orientation == ImageOrientation::Orientation::FromImage)
            orientation = currentFrameOrientation();

        auto headroom = options.headroom();
        if (headroom == Headroom::FromImage)
            headroom = currentFrameHeadroom();

        context.drawNativeImage(*nativeImage, destinationRect, adjustedSourceRect, { options, orientation, headroom });
    }

    if (auto observer = imageObserver())
        observer->didDraw(*this);

    return ImageDrawResult::DidDraw;
}

void BitmapImage::drawPattern(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& tileRect, const AffineTransform& transform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    if (tileRect.isEmpty())
        return;

    if (context.drawLuminanceMask())
        drawLuminanceMaskPattern(context, destinationRect, tileRect, transform, phase, spacing, options);
    else
        Image::drawPattern(context, destinationRect, tileRect, transform, phase, spacing, { options, ImageOrientation::Orientation::FromImage });
}

void BitmapImage::drawLuminanceMaskPattern(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& tileRect, const AffineTransform& transform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    ASSERT(!tileRect.isEmpty());
    ASSERT(context.drawLuminanceMask());

    auto buffer = context.createImageBuffer(expandedIntSize(tileRect.size()));
    if (!buffer)
        return;

    auto observer = imageObserver();

    // Temporarily reset image observer, we don't want to receive any changeInRect() calls due to this relayout.
    setImageObserver(nullptr);

    auto bufferRect = FloatRect { { }, buffer->logicalSize() };
    draw(buffer->context(), bufferRect, tileRect, { options, DecodingMode::Synchronous, ImageOrientation::Orientation::FromImage });

    setImageObserver(WTFMove(observer));
    buffer->convertToLuminanceMask();

    context.setDrawLuminanceMask(false);
    context.drawPattern(*buffer, destinationRect, bufferRect, transform, phase, spacing, { options, ImageOrientation::Orientation::FromImage });
}

void BitmapImage::dump(TextStream& ts) const
{
    Image::dump(ts);
    m_source->dump(ts);
}

} // namespace WebCore
