/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "SVGResourceImage.h"

#include "LegacyRenderSVGResourceMasker.h"
#include "ReferencedSVGResources.h"
#include "RenderSVGResourceMasker.h"

namespace WebCore {

Ref<SVGResourceImage> SVGResourceImage::create(RenderSVGResourceContainer& renderResource, const URL& reresolvedURL)
{
    return adoptRef(*new SVGResourceImage(renderResource, reresolvedURL));
}

SVGResourceImage::SVGResourceImage(RenderSVGResourceContainer& renderResource, const URL& reresolvedURL)
    : m_renderResource(renderResource)
    , m_reresolvedURL(reresolvedURL)
{
}

Ref<SVGResourceImage> SVGResourceImage::create(LegacyRenderSVGResourceContainer& renderResource, const URL& reresolvedURL)
{
    return adoptRef(*new SVGResourceImage(renderResource, reresolvedURL));
}

SVGResourceImage::SVGResourceImage(LegacyRenderSVGResourceContainer& renderResource, const URL& reresolvedURL)
    : m_legacyRenderResource(renderResource)
    , m_reresolvedURL(reresolvedURL)
{
}

ImageDrawResult SVGResourceImage::draw(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    if (CheckedPtr masker = dynamicDowncast<RenderSVGResourceMasker>(m_renderResource.get())) {
        if (masker->drawContentIntoContext(context, destinationRect, sourceRect, options))
            return ImageDrawResult::DidDraw;
    }
    if (CheckedPtr masker = dynamicDowncast<LegacyRenderSVGResourceMasker>(m_legacyRenderResource.get())) {
        if (masker->drawContentIntoContext(context, destinationRect, sourceRect, options))
            return ImageDrawResult::DidDraw;
    }

    return ImageDrawResult::DidNothing;
}

void SVGResourceImage::drawPattern(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions options)
{
    RefPtr imageBuffer = context.createImageBuffer(size());
    if (!imageBuffer)
        return;

    // Fill with the SVG resource.
    GraphicsContext& graphicsContext = imageBuffer->context();
    graphicsContext.drawImage(*this, FloatPoint());

    // Tile the image buffer into the context.
    context.drawPattern(*imageBuffer, destinationRect, sourceRect, patternTransform, phase, spacing, options);
}

void SVGResourceImage::dump(WTF::TextStream& ts) const
{
    GeneratedImage::dump(ts);
    ts << m_reresolvedURL;
}

} // namespace WebCore
