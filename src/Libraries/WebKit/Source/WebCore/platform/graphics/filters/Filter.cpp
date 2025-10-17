/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include "Filter.h"

#include "FilterEffect.h"
#include "FilterImage.h"
#include "FilterResults.h"
#include "FilterStyle.h"
#include "ImageBuffer.h"

namespace WebCore {

Filter::Filter(Filter::Type filterType, std::optional<RenderingResourceIdentifier> renderingResourceIdentifier)
    : FilterFunction(filterType, renderingResourceIdentifier)
{
}

Filter::Filter(Filter::Type filterType, const FloatSize& filterScale, const FloatRect& filterRegion, std::optional<RenderingResourceIdentifier> renderingResourceIdentifier)
    : FilterFunction(filterType, renderingResourceIdentifier)
    , m_filterScale(filterScale)
    , m_filterRegion(filterRegion)
{
}

FloatPoint Filter::scaledByFilterScale(const FloatPoint& point) const
{
    return point.scaled(m_filterScale.width(), m_filterScale.height());
}

FloatSize Filter::scaledByFilterScale(const FloatSize& size) const
{
    return size * m_filterScale;
}

FloatRect Filter::scaledByFilterScale(const FloatRect& rect) const
{
    auto scaledRect = rect;
    scaledRect.scale(m_filterScale);
    return scaledRect;
}

FloatRect Filter::maxEffectRect(const FloatRect& primitiveSubregion) const
{
    return intersection(primitiveSubregion, m_filterRegion);
}

FloatRect Filter::clipToMaxEffectRect(const FloatRect& imageRect, const FloatRect& primitiveSubregion) const
{
    auto maxEffectRect = this->maxEffectRect(primitiveSubregion);
    return intersection(imageRect, maxEffectRect);
}

bool Filter::clampFilterRegionIfNeeded()
{
    auto scaledFilterRegion = scaledByFilterScale(m_filterRegion);

    FloatSize clampingScale(1, 1);
    if (!ImageBuffer::sizeNeedsClamping(scaledFilterRegion.size(), clampingScale))
        return false;

    m_filterScale = m_filterScale * clampingScale;
    return true;
}

RenderingMode Filter::renderingMode() const
{
    if (m_filterRenderingModes.contains(FilterRenderingMode::Accelerated))
        return RenderingMode::Accelerated;
    
    ASSERT(m_filterRenderingModes.contains(FilterRenderingMode::Software));
    return RenderingMode::Unaccelerated;
}

void Filter::setFilterRenderingModes(OptionSet<FilterRenderingMode> preferredFilterRenderingModes)
{
    m_filterRenderingModes = preferredFilterRenderingModes & supportedFilterRenderingModes();
    ASSERT(m_filterRenderingModes.contains(FilterRenderingMode::Software));
}

RefPtr<FilterImage> Filter::apply(ImageBuffer* sourceImage, const FloatRect& sourceImageRect, FilterResults& results)
{
    RefPtr<FilterImage> input;

    if (sourceImage) {
        auto absoluteSourceImageRect = enclosingIntRect(scaledByFilterScale(sourceImageRect));
        input = FilterImage::create(m_filterRegion, sourceImageRect, absoluteSourceImageRect, Ref { *sourceImage }, results.allocator());
        if (!input)
            return nullptr;
    }

    auto result = apply(input.get(), results);
    if (!result)
        return nullptr;

    result->correctPremultipliedPixelBuffer();
    result->transformToColorSpace(DestinationColorSpace::SRGB());
    return result;
}

FilterStyleVector Filter::createFilterStyles(GraphicsContext& context, const FloatRect& sourceImageRect) const
{
    auto input = FilterStyle { std::nullopt, m_filterRegion, sourceImageRect };
    auto result = createFilterStyles(context, input);
    if (result.isEmpty())
        return { };

    result.reverse();
    result.shrinkToFit();
    return result;
}

} // namespace WebCore
