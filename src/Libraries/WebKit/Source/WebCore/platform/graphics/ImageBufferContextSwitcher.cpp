/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#include "ImageBufferContextSwitcher.h"

#include "Filter.h"
#include "FilterResults.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageBufferContextSwitcher);

ImageBufferContextSwitcher::ImageBufferContextSwitcher(GraphicsContext& destinationContext, const FloatRect& sourceImageRect, const DestinationColorSpace& colorSpace, RefPtr<Filter>&& filter, FilterResults* results)
    : GraphicsContextSwitcher(WTFMove(filter))
    , m_sourceImageRect(sourceImageRect)
    , m_results(results)
{
    if (sourceImageRect.isEmpty())
        return;

    if (m_filter)
        m_sourceImage = destinationContext.createScaledImageBuffer(m_sourceImageRect, m_filter->filterScale(), colorSpace, m_filter->renderingMode());
    else
        m_sourceImage = destinationContext.createAlignedImageBuffer(m_sourceImageRect, colorSpace);

    if (!m_sourceImage) {
        m_filter = nullptr;
        return;
    }

    auto state = destinationContext.state();
    m_sourceImage->context().mergeAllChanges(state);
}

GraphicsContext* ImageBufferContextSwitcher::drawingContext(GraphicsContext& context) const
{
    return m_sourceImage ? &m_sourceImage->context() : &context;
}

void ImageBufferContextSwitcher::beginClipAndDrawSourceImage(GraphicsContext& destinationContext, const FloatRect& repaintRect, const FloatRect&)
{
    if (auto* context = drawingContext(destinationContext)) {
        context->save();
        context->clearRect(repaintRect);
        context->clip(repaintRect);
    }
}

void ImageBufferContextSwitcher::endClipAndDrawSourceImage(GraphicsContext& destinationContext, const DestinationColorSpace& colorSpace)
{
    if (auto* context = drawingContext(destinationContext))
        context->restore();

    endDrawSourceImage(destinationContext, colorSpace);
}

void ImageBufferContextSwitcher::endDrawSourceImage(GraphicsContext& destinationContext, const DestinationColorSpace& colorSpace)
{
    if (!m_filter) {
        if (m_sourceImage)
            destinationContext.drawImageBuffer(*m_sourceImage, m_sourceImageRect, { destinationContext.compositeOperation(), destinationContext.blendMode() });
        return;
    }

    FilterResults results;
#if USE(CAIRO)
    // Cairo operates in SRGB which is why the SourceImage initially is in SRGB color space,
    // but before applying all filters it has to be transformed to LinearRGB to comply with
    // specification (https://www.w3.org/TR/filter-effects-1/#attr-valuedef-in-sourcegraphic).
    if (m_sourceImage)
        m_sourceImage->transformToColorSpace(colorSpace);
#else
    UNUSED_PARAM(colorSpace);
#endif
    destinationContext.drawFilteredImageBuffer(m_sourceImage.get(), m_sourceImageRect, Ref { *m_filter }, m_results ? *m_results : results);
}

} // namespace WebCore
