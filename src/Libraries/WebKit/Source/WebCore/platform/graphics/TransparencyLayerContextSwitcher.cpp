/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include "TransparencyLayerContextSwitcher.h"

#include "Filter.h"
#include "GraphicsContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TransparencyLayerContextSwitcher);

TransparencyLayerContextSwitcher::TransparencyLayerContextSwitcher(GraphicsContext& destinationContext, const FloatRect& sourceImageRect, RefPtr<Filter>&& filter)
    : GraphicsContextSwitcher(WTFMove(filter))
{
    if (m_filter)
        m_filterStyles = m_filter->createFilterStyles(destinationContext, sourceImageRect);
}

void TransparencyLayerContextSwitcher::beginClipAndDrawSourceImage(GraphicsContext& destinationContext, const FloatRect&, const FloatRect& clipRect)
{
    destinationContext.save();
    destinationContext.beginTransparencyLayer(1);

    for (auto& filterStyle : m_filterStyles) {
        destinationContext.save();
        destinationContext.clip(intersection(filterStyle.imageRect, clipRect));
        destinationContext.setStyle(filterStyle.style);
        destinationContext.beginTransparencyLayer(1);
    }
}

void TransparencyLayerContextSwitcher::beginDrawSourceImage(GraphicsContext& destinationContext, float opacity)
{
    destinationContext.save();
    destinationContext.beginTransparencyLayer(opacity);

    for (auto& filterStyle : m_filterStyles) {
        destinationContext.save();
        destinationContext.clip(filterStyle.imageRect);
        destinationContext.setStyle(filterStyle.style);
        destinationContext.beginTransparencyLayer(1);
    }
}

void TransparencyLayerContextSwitcher::endDrawSourceImage(GraphicsContext& destinationContext, const DestinationColorSpace&)
{
    for ([[maybe_unused]] auto& filterStyle : makeReversedRange(m_filterStyles)) {
        destinationContext.endTransparencyLayer();
        destinationContext.restore();
    }

    destinationContext.endTransparencyLayer();
    destinationContext.restore();
}

} // namespace WebCore
