/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#include "LegacyRenderSVGViewportContainer.h"

#include "GraphicsContext.h"
#include "RenderView.h"
#include "SVGElementTypeHelpers.h"
#include "SVGSVGElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGViewportContainer);

LegacyRenderSVGViewportContainer::LegacyRenderSVGViewportContainer(SVGSVGElement& element, RenderStyle&& style)
    : LegacyRenderSVGContainer(Type::LegacySVGViewportContainer, element, WTFMove(style))
    , m_didTransformToRootUpdate(false)
    , m_isLayoutSizeChanged(false)
    , m_needsTransformUpdate(true)
{
    ASSERT(isLegacyRenderSVGViewportContainer());
}

LegacyRenderSVGViewportContainer::~LegacyRenderSVGViewportContainer() = default;

SVGSVGElement& LegacyRenderSVGViewportContainer::svgSVGElement() const
{
    return downcast<SVGSVGElement>(LegacyRenderSVGContainer::element());
}

void LegacyRenderSVGViewportContainer::determineIfLayoutSizeChanged()
{
    m_isLayoutSizeChanged = svgSVGElement().hasRelativeLengths() && selfNeedsLayout();
}

void LegacyRenderSVGViewportContainer::applyViewportClip(PaintInfo& paintInfo)
{
    if (SVGRenderSupport::isOverflowHidden(*this))
        paintInfo.context().clip(m_viewport);
}

void LegacyRenderSVGViewportContainer::calcViewport()
{
    Ref element = svgSVGElement();
    SVGLengthContext lengthContext(element.ptr());
    FloatRect newViewport(element->x().value(lengthContext), element->y().value(lengthContext), element->width().value(lengthContext), element->height().value(lengthContext));

    if (m_viewport == newViewport)
        return;

    m_viewport = newViewport;

    invalidateCachedBoundaries();
    setNeedsTransformUpdate();
}

bool LegacyRenderSVGViewportContainer::calculateLocalTransform()
{
    m_didTransformToRootUpdate = m_needsTransformUpdate || SVGRenderSupport::transformToRootChanged(parent());
    if (!m_needsTransformUpdate)
        return false;

    m_localToParentTransform = AffineTransform::makeTranslation(toFloatSize(m_viewport.location())) * viewportTransform();
    m_needsTransformUpdate = false;
    return true;
}

AffineTransform LegacyRenderSVGViewportContainer::viewportTransform() const
{
    return svgSVGElement().viewBoxToViewTransform(m_viewport.width(), m_viewport.height());
}

bool LegacyRenderSVGViewportContainer::pointIsInsideViewportClip(const FloatPoint& pointInParent)
{
    // Respect the viewport clip (which is in parent coords)
    if (!SVGRenderSupport::isOverflowHidden(*this))
        return true;

    return m_viewport.contains(pointInParent);
}

void LegacyRenderSVGViewportContainer::paint(PaintInfo& paintInfo, const LayoutPoint& paintOffset)
{
    // An empty viewBox disables rendering.
    if (svgSVGElement().hasEmptyViewBox())
        return;

    LegacyRenderSVGContainer::paint(paintInfo, paintOffset);
}

}
