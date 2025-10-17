/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#include "CanvasLayerContextSwitcher.h"

#include "CanvasRenderingContext2DBase.h"
#include "Filter.h"
#include "GraphicsContextSwitcher.h"

namespace WebCore {

RefPtr<CanvasLayerContextSwitcher> CanvasLayerContextSwitcher::create(CanvasRenderingContext2DBase& context, const FloatRect& bounds, RefPtr<Filter>&& filter)
{
    ASSERT(!bounds.isEmpty());
    auto* effectiveDrawingContext = context.effectiveDrawingContext();
    if (!effectiveDrawingContext)
        return nullptr;

    auto targetSwitcher = GraphicsContextSwitcher::create(*effectiveDrawingContext, bounds, context.colorSpace(), WTFMove(filter));
    if (!targetSwitcher)
        return nullptr;

    return adoptRef(*new CanvasLayerContextSwitcher(context, bounds, WTFMove(targetSwitcher)));
}

CanvasLayerContextSwitcher::CanvasLayerContextSwitcher(CanvasRenderingContext2DBase& context, const FloatRect& bounds, std::unique_ptr<GraphicsContextSwitcher>&& targetSwitcher)
    : m_context(context)
    , m_effectiveDrawingContext(context.effectiveDrawingContext())
    , m_bounds(bounds)
    , m_targetSwitcher(WTFMove(targetSwitcher))
{
    ASSERT(m_targetSwitcher);
    ASSERT(m_effectiveDrawingContext);
    m_targetSwitcher->beginDrawSourceImage(*m_effectiveDrawingContext, context.globalAlpha());
}

CanvasLayerContextSwitcher::~CanvasLayerContextSwitcher()
{
    m_targetSwitcher->endDrawSourceImage(*m_effectiveDrawingContext, DestinationColorSpace::SRGB());
}

GraphicsContext* CanvasLayerContextSwitcher::drawingContext() const
{
    if (!m_effectiveDrawingContext)
        return nullptr;
    return m_targetSwitcher->drawingContext(*m_effectiveDrawingContext);
}

FloatBoxExtent CanvasLayerContextSwitcher::outsets() const
{
    return toFloatBoxExtent(protectedContext()->calculateFilterOutsets(m_bounds));
}

} // namespace WebCore
