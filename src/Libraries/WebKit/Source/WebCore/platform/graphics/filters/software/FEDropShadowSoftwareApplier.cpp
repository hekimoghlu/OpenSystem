/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
#include "FEDropShadowSoftwareApplier.h"

#include "FEDropShadow.h"
#include "Filter.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "PixelBuffer.h"
#include "ShadowBlur.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEDropShadowSoftwareApplier);

bool FEDropShadowSoftwareApplier::apply(const Filter& filter, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();

    RefPtr resultImage = result.imageBuffer();
    if (!resultImage)
        return false;

    auto stdDeviation = filter.resolvedSize({ m_effect.stdDeviationX(), m_effect.stdDeviationY() });
    auto blurRadius = 2 * filter.scaledByFilterScale(stdDeviation);
    
    auto offset = filter.resolvedSize({ m_effect.dx(), m_effect.dy() });
    auto absoluteOffset = filter.scaledByFilterScale(offset);

    FloatRect inputImageRect = input.absoluteImageRectRelativeTo(result);
    FloatRect inputImageRectWithOffset(inputImageRect);
    inputImageRectWithOffset.move(absoluteOffset);

    RefPtr inputImage = input.imageBuffer();
    if (!inputImage)
        return false;

    auto& resultContext = resultImage->context();
    resultContext.setAlpha(m_effect.shadowOpacity());
    resultContext.drawImageBuffer(*inputImage, inputImageRectWithOffset);
    resultContext.setAlpha(1);

    ShadowBlur contextShadow(blurRadius, absoluteOffset, m_effect.shadowColor());

    PixelBufferFormat format { AlphaPremultiplication::Premultiplied, PixelFormat::RGBA8, result.colorSpace() };
    IntRect shadowArea(IntPoint(), resultImage->truncatedLogicalSize());
    auto pixelBuffer = resultImage->getPixelBuffer(format, shadowArea);
    if (!pixelBuffer)
        return false;

    contextShadow.blurLayerImage(pixelBuffer->bytes(), pixelBuffer->size(), 4 * pixelBuffer->size().width());

    resultImage->putPixelBuffer(*pixelBuffer, shadowArea);

    resultContext.setCompositeOperation(CompositeOperator::SourceIn);
    resultContext.fillRect(FloatRect(FloatPoint(), result.absoluteImageRect().size()), m_effect.shadowColor());
    resultContext.setCompositeOperation(CompositeOperator::DestinationOver);

    resultImage->context().drawImageBuffer(*inputImage, inputImageRect);
    return true;
}

} // namespace WebCore
