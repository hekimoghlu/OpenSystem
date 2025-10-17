/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#include "GradientImage.h"

#include "Gradient.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"

namespace WebCore {

GradientImage::GradientImage(Gradient& generator, const FloatSize& size)
    : m_gradient(generator)
{
    setContainerSize(size);
}

GradientImage::~GradientImage() = default;

ImageDrawResult GradientImage::draw(GraphicsContext& destContext, const FloatRect& destRect, const FloatRect& srcRect, ImagePaintingOptions options)
{
    GraphicsContextStateSaver stateSaver(destContext);
    destContext.setCompositeOperation(options.compositeOperator(), options.blendMode());
    destContext.clip(destRect);
    destContext.translate(destRect.location());
    if (destRect.size() != srcRect.size())
        destContext.scale(destRect.size() / srcRect.size());
    destContext.translate(-srcRect.location());
    destContext.fillRect(FloatRect(FloatPoint(), size()), m_gradient.get());
    return ImageDrawResult::DidDraw;
}

void GradientImage::drawPattern(GraphicsContext& destContext, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform& patternTransform,
    const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    // Allow the generator to provide visually-equivalent tiling parameters for better performance.
    FloatSize adjustedSize = size();
    FloatRect adjustedSrcRect = srcRect;
    m_gradient->adjustParametersForTiledDrawing(adjustedSize, adjustedSrcRect, spacing);

    // Factor in the destination context's scale to generate at the best resolution
    AffineTransform destContextCTM = destContext.getCTM(GraphicsContext::DefinitelyIncludeDeviceScale);
    double xScale = std::abs(destContextCTM.xScale());
    double yScale = std::abs(destContextCTM.yScale());
    AffineTransform adjustedPatternCTM = patternTransform;
    adjustedPatternCTM.scale(1.0 / xScale, 1.0 / yScale);
    adjustedSrcRect.scale(xScale, yScale);

    unsigned generatorHash = m_gradient->hash();

    if (!m_cachedImage || m_cachedGeneratorHash != generatorHash || m_cachedAdjustedSize != adjustedSize || !areEssentiallyEqual(destContext.scaleFactor(), m_cachedScaleFactor)) {
        auto imageBuffer = destContext.createAlignedImageBuffer(adjustedSize);
        if (!imageBuffer)
            return;

        // Fill with the generated image.
        imageBuffer->context().fillRect(FloatRect(FloatPoint(), adjustedSize), m_gradient.get());

        m_cachedGeneratorHash = generatorHash;
        m_cachedAdjustedSize = adjustedSize;
        m_cachedScaleFactor = destContext.scaleFactor();

        if (destContext.drawLuminanceMask())
            imageBuffer->convertToLuminanceMask();

        m_cachedImage = WTFMove(imageBuffer);
        if (!m_cachedImage)
            return;
    }

    destContext.setDrawLuminanceMask(false);
    destContext.drawPattern(Ref { *m_cachedImage }, destRect, adjustedSrcRect, adjustedPatternCTM, phase, spacing, options);

}

void GradientImage::dump(WTF::TextStream& ts) const
{
    GeneratedImage::dump(ts);
    // FIXME: dump the gradient.
}

}
