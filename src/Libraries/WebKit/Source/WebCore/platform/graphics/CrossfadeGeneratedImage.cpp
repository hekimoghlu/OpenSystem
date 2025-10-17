/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#include "CrossfadeGeneratedImage.h"

#include "FloatRect.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

CrossfadeGeneratedImage::CrossfadeGeneratedImage(Image& fromImage, Image& toImage, float percentage, const FloatSize& crossfadeSize, const FloatSize& size)
    : m_fromImage(fromImage)
    , m_toImage(toImage)
    , m_percentage(percentage)
    , m_crossfadeSize(crossfadeSize)
{
    setContainerSize(size);
}

static void drawCrossfadeSubimage(GraphicsContext& context, Image& image, CompositeOperator operation, float opacity, const FloatSize& targetSize)
{
    FloatSize imageSize = image.size();

    // SVGImage resets the opacity when painting, so we have to use transparency layers to accurately paint one at a given opacity.
    bool useTransparencyLayer = image.drawsSVGImage();

    GraphicsContextStateSaver stateSaver(context);

    ImagePaintingOptions options;

    if (useTransparencyLayer) {
        context.setCompositeOperation(operation);
        context.beginTransparencyLayer(opacity);
    } else {
        context.setAlpha(opacity);
        options = { operation };
    }

    if (targetSize != imageSize)
        context.scale(targetSize / imageSize);

    context.drawImage(image, IntPoint(), options);

    if (useTransparencyLayer)
        context.endTransparencyLayer();
}

void CrossfadeGeneratedImage::drawCrossfade(GraphicsContext& context)
{
    // Draw nothing if either of the images hasn't loaded yet.
    if (m_fromImage.ptr() == &Image::nullImage() || m_toImage.ptr() == &Image::nullImage())
        return;

    GraphicsContextStateSaver stateSaver(context);

    context.clip(FloatRect(FloatPoint(), m_crossfadeSize));
    context.beginTransparencyLayer(1);

    drawCrossfadeSubimage(context, m_fromImage.get(), CompositeOperator::SourceOver, 1 - m_percentage, m_crossfadeSize);
    drawCrossfadeSubimage(context, m_toImage.get(), CompositeOperator::PlusLighter, m_percentage, m_crossfadeSize);

    context.endTransparencyLayer();
}

ImageDrawResult CrossfadeGeneratedImage::draw(GraphicsContext& context, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions options)
{
    GraphicsContextStateSaver stateSaver(context);
    context.setCompositeOperation(options.compositeOperator(), options.blendMode());
    context.clip(dstRect);
    context.translate(dstRect.location());
    if (dstRect.size() != srcRect.size())
        context.scale(dstRect.size() / srcRect.size());
    context.translate(-srcRect.location());
    
    drawCrossfade(context);
    return ImageDrawResult::DidDraw;
}

void CrossfadeGeneratedImage::drawPattern(GraphicsContext& context, const FloatRect& dstRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    auto imageBuffer = context.createImageBuffer(size());
    if (!imageBuffer)
        return;

    // Fill with the cross-faded image.
    GraphicsContext& graphicsContext = imageBuffer->context();
    drawCrossfade(graphicsContext);

    // Tile the image buffer into the context.
    context.drawPattern(*imageBuffer, dstRect, srcRect, patternTransform, phase, spacing, options);
}

void CrossfadeGeneratedImage::dump(TextStream& ts) const
{
    GeneratedImage::dump(ts);
    ts.dumpProperty("from-image", m_fromImage.get());
    ts.dumpProperty("to-image", m_toImage.get());
    ts.dumpProperty("percentage", m_percentage);
}

}
