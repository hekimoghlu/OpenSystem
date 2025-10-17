/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#pragma once

#include "FilterEffectVector.h"
#include "FilterFunction.h"
#include "FloatPoint3D.h"
#include "FloatRect.h"
#include "GraphicsTypes.h"
#include "ImageBuffer.h"
#include "RenderingMode.h"

namespace WebCore {

class FilterEffect;
class FilterImage;
class FilterResults;

class Filter : public FilterFunction {
    using FilterFunction::apply;
    using FilterFunction::createFilterStyles;

public:
    RenderingMode renderingMode() const;

    OptionSet<FilterRenderingMode> filterRenderingModes() const { return m_filterRenderingModes; }
    WEBCORE_EXPORT void setFilterRenderingModes(OptionSet<FilterRenderingMode> preferredFilterRenderingModes);

    FloatSize filterScale() const { return m_filterScale; }
    void setFilterScale(const FloatSize& filterScale) { m_filterScale = filterScale; }

    FloatRect filterRegion() const { return m_filterRegion; }
    void setFilterRegion(const FloatRect& filterRegion) { m_filterRegion = filterRegion; }

    virtual FloatSize resolvedSize(const FloatSize& size) const { return size; }
    virtual FloatPoint3D resolvedPoint3D(const FloatPoint3D& point) const { return point; }

    FloatPoint scaledByFilterScale(const FloatPoint&) const;
    FloatSize scaledByFilterScale(const FloatSize&) const;
    FloatRect scaledByFilterScale(const FloatRect&) const;

    FloatRect maxEffectRect(const FloatRect& primitiveSubregion) const;
    FloatRect clipToMaxEffectRect(const FloatRect& imageRect, const FloatRect& primitiveSubregion) const;

    virtual FilterEffectVector effectsOfType(FilterFunction::Type) const = 0;

    bool clampFilterRegionIfNeeded();

    WEBCORE_EXPORT RefPtr<FilterImage> apply(ImageBuffer* sourceImage, const FloatRect& sourceImageRect, FilterResults&);
    WEBCORE_EXPORT FilterStyleVector createFilterStyles(GraphicsContext&, const FloatRect& sourceImageRect) const;

protected:
    Filter(Filter::Type, std::optional<RenderingResourceIdentifier> = std::nullopt);
    Filter(Filter::Type, const FloatSize& filterScale, const FloatRect& filterRegion = { }, std::optional<RenderingResourceIdentifier> = std::nullopt);

    virtual RefPtr<FilterImage> apply(FilterImage* sourceImage, FilterResults&) = 0;
    virtual FilterStyleVector createFilterStyles(GraphicsContext&, const FilterStyle& sourceStyle) const = 0;

private:
    OptionSet<FilterRenderingMode> m_filterRenderingModes { FilterRenderingMode::Software };
    FloatSize m_filterScale;
    FloatRect m_filterRegion;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Filter)
    static bool isType(const WebCore::RenderingResource& renderingResource) { return renderingResource.isFilter(); }
SPECIALIZE_TYPE_TRAITS_END()
