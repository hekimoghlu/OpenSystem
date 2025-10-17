/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#include "CairoPaintingEngineBasic.h"

#if USE(CAIRO)
#include "CairoPaintingContext.h"
#include "CoordinatedTileBuffer.h"
#include "GraphicsContext.h"
#include "GraphicsLayer.h"

namespace WebCore {
namespace Cairo {

PaintingEngineBasic::PaintingEngineBasic() = default;
PaintingEngineBasic::~PaintingEngineBasic() = default;

void PaintingEngineBasic::paint(GraphicsLayer& layer, CoordinatedTileBuffer& buffer, const IntRect& sourceRect, const IntRect& mappedSourceRect, const IntRect& targetRect, float contentsScale)
{
    buffer.beginPainting();

    bool supportsAlpha = buffer.supportsAlpha();
    PaintingContext::paint(buffer,
        [&layer, sourceRect, mappedSourceRect, targetRect, contentsScale, supportsAlpha]
        (GraphicsContext& context)
        {
            context.save();
            context.clip(targetRect);
            context.translate(targetRect.x(), targetRect.y());

            if (supportsAlpha) {
                context.setCompositeOperation(CompositeOperator::Copy);
                context.fillRect(IntRect(IntPoint::zero(), sourceRect.size()), Color::transparentBlack);
                context.setCompositeOperation(CompositeOperator::SourceOver);
            }

            context.translate(-sourceRect.x(), -sourceRect.y());
            context.scale(FloatSize(contentsScale, contentsScale));

            layer.paintGraphicsLayerContents(context, mappedSourceRect);

            context.restore();
        });

    buffer.completePainting();
}

} // namespace Cairo
} // namespace WebCore

#endif // USE(CAIRO)
