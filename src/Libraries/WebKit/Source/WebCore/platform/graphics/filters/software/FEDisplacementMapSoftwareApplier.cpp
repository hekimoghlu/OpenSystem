/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
#include "FEDisplacementMapSoftwareApplier.h"

#include "FEDisplacementMap.h"
#include "Filter.h"
#include "GraphicsContext.h"
#include "PixelBuffer.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FEDisplacementMapSoftwareApplier);

FEDisplacementMapSoftwareApplier::FEDisplacementMapSoftwareApplier(const FEDisplacementMap& effect)
    : Base(effect)
{
    ASSERT(m_effect.xChannelSelector() != ChannelSelectorType::CHANNEL_UNKNOWN);
    ASSERT(m_effect.yChannelSelector() != ChannelSelectorType::CHANNEL_UNKNOWN);
}

int FEDisplacementMapSoftwareApplier::xChannelIndex() const
{
    return static_cast<int>(m_effect.xChannelSelector()) - 1;
}

int FEDisplacementMapSoftwareApplier::yChannelIndex() const
{
    return static_cast<int>(m_effect.yChannelSelector()) - 1;
}

bool FEDisplacementMapSoftwareApplier::apply(const Filter& filter, const FilterImageVector& inputs, FilterImage& result) const
{
    auto& input = inputs[0].get();
    auto& input2 = inputs[1].get();

    auto destinationPixelBuffer = result.pixelBuffer(AlphaPremultiplication::Premultiplied);
    if (!destinationPixelBuffer)
        return false;

    auto effectADrawingRect = result.absoluteImageRectRelativeTo(input);
    auto inputPixelBuffer = input.getPixelBuffer(AlphaPremultiplication::Premultiplied, effectADrawingRect);

    auto effectBDrawingRect = result.absoluteImageRectRelativeTo(input2);
    // The calculations using the pixel values from â€˜in2â€™ are performed using non-premultiplied color values.
    auto displacementPixelBuffer = input2.getPixelBuffer(AlphaPremultiplication::Unpremultiplied, effectBDrawingRect);
    
    if (!inputPixelBuffer || !displacementPixelBuffer)
        return false;

    ASSERT(inputPixelBuffer->bytes().size() == displacementPixelBuffer->bytes().size());

    auto paintSize = result.absoluteImageRect().size();
    auto scale = filter.resolvedSize({ m_effect.scale(), m_effect.scale() });
    auto absoluteScale = filter.scaledByFilterScale(scale);

    float scaleForColorX = absoluteScale.width() / 255.0;
    float scaleForColorY = absoluteScale.height() / 255.0;
    float scaledOffsetX = 0.5 - absoluteScale.width() * 0.5;
    float scaledOffsetY = 0.5 - absoluteScale.height() * 0.5;
    
    int displacementChannelX = xChannelIndex();
    int displacementChannelY = yChannelIndex();

    int rowBytes = paintSize.width() * 4;

    for (int y = 0; y < paintSize.height(); ++y) {
        int lineStartOffset = y * rowBytes;

        for (int x = 0; x < paintSize.width(); ++x) {
            int destinationIndex = lineStartOffset + x * 4;
            
            int srcX = x + static_cast<int>(scaleForColorX * displacementPixelBuffer->item(destinationIndex + displacementChannelX) + scaledOffsetX);
            int srcY = y + static_cast<int>(scaleForColorY * displacementPixelBuffer->item(destinationIndex + displacementChannelY) + scaledOffsetY);

            unsigned& destinationPixel = reinterpretCastSpanStartTo<unsigned>(destinationPixelBuffer->bytes().subspan(destinationIndex));
            if (srcX < 0 || srcX >= paintSize.width() || srcY < 0 || srcY >= paintSize.height()) {
                destinationPixel = 0;
                continue;
            }

            destinationPixel = reinterpretCastSpanStartTo<unsigned>(inputPixelBuffer->bytes().subspan(byteOffsetOfPixel(srcX, srcY, rowBytes)));
        }
    }

    return true;
}

} // namespace WebCore
