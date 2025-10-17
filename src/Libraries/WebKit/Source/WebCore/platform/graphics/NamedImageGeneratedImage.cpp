/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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
#include "NamedImageGeneratedImage.h"

#include "FloatRect.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "Theme.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

NamedImageGeneratedImage::NamedImageGeneratedImage(String name, const FloatSize& size)
    : m_name(name)
{
    setContainerSize(size);
}

ImageDrawResult NamedImageGeneratedImage::draw(GraphicsContext& context, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions options)
{
    GraphicsContextStateSaver stateSaver(context);
    context.setCompositeOperation(options.compositeOperator(), options.blendMode());
    context.clip(dstRect);
    context.translate(dstRect.location());
    if (dstRect.size() != srcRect.size())
        context.scale(FloatSize(dstRect.width() / srcRect.width(), dstRect.height() / srcRect.height()));
    context.translate(-srcRect.location());

    Theme::singleton().drawNamedImage(m_name, context, dstRect.size());
    return ImageDrawResult::DidDraw;
}

void NamedImageGeneratedImage::drawPattern(GraphicsContext& context, const FloatRect& dstRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    auto imageBuffer = context.createAlignedImageBuffer(size());
    if (!imageBuffer)
        return;

    GraphicsContext& graphicsContext = imageBuffer->context();
    Theme::singleton().drawNamedImage(m_name, graphicsContext, size());

    // Tile the image buffer into the context.
    context.drawPattern(*imageBuffer, dstRect, srcRect, patternTransform, phase, spacing, options);
}

void NamedImageGeneratedImage::dump(TextStream& ts) const
{
    GeneratedImage::dump(ts);
    ts.dumpProperty("name", m_name);
}

}
