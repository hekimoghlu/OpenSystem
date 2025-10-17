/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include "RenderSVGInline.h"

#include "LegacyRenderSVGResource.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderSVGInlineInlines.h"
#include "RenderSVGInlineText.h"
#include "RenderSVGText.h"
#include "SVGGraphicsElement.h"
#include "SVGInlineFlowBox.h"
#include "SVGRenderSupport.h"
#include "SVGResourcesCache.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGInline);
    
RenderSVGInline::RenderSVGInline(Type type, SVGGraphicsElement& element, RenderStyle&& style)
    : RenderInline(type, element, WTFMove(style))
{
    ASSERT(isRenderSVGInline());
}

RenderSVGInline::~RenderSVGInline() = default;

std::unique_ptr<LegacyInlineFlowBox> RenderSVGInline::createInlineFlowBox()
{
    auto box = makeUnique<SVGInlineFlowBox>(*this);
    box->setHasVirtualLogicalHeight();
    return box;
}

bool RenderSVGInline::isChildAllowed(const RenderObject& child, const RenderStyle& style) const
{
    auto isEmptySVGInlineText = [](const RenderObject* object) {
        const auto svgInlineText = dynamicDowncast<RenderSVGInlineText>(object);
        return svgInlineText && svgInlineText->hasEmptyText();
    };

    if (isEmptySVGInlineText(&child))
        return false;

    return RenderElement::isChildAllowed(child, style);
}

FloatRect RenderSVGInline::objectBoundingBox() const
{
    if (auto* textAncestor = RenderSVGText::locateRenderSVGTextAncestor(*this))
        return textAncestor->objectBoundingBox();

    return FloatRect();
}

FloatRect RenderSVGInline::strokeBoundingBox() const
{
    if (auto* textAncestor = RenderSVGText::locateRenderSVGTextAncestor(*this))
        return textAncestor->strokeBoundingBox();

    return FloatRect();
}

FloatRect RenderSVGInline::repaintRectInLocalCoordinates(RepaintRectCalculation repaintRectCalculation) const
{
    if (auto* textAncestor = RenderSVGText::locateRenderSVGTextAncestor(*this))
        return textAncestor->repaintRectInLocalCoordinates(repaintRectCalculation);

    return FloatRect();
}

LayoutRect RenderSVGInline::clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext context) const
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return RenderInline::clippedOverflowRect(repaintContainer, context);
    return SVGRenderSupport::clippedOverflowRectForRepaint(*this, repaintContainer, context);
}

auto RenderSVGInline::rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds repaintOutlineBounds) const -> RepaintRects
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return RenderInline::rectsForRepaintingAfterLayout(repaintContainer, repaintOutlineBounds);

    auto rects = RepaintRects { SVGRenderSupport::clippedOverflowRectForRepaint(*this, repaintContainer, visibleRectContextForRepaint()) };
    if (repaintOutlineBounds == RepaintOutlineBounds::Yes)
        rects.outlineBoundsRect = outlineBoundsForRepaint(repaintContainer);

    return rects;
}

std::optional<FloatRect> RenderSVGInline::computeFloatVisibleRectInContainer(const FloatRect& rect, const RenderLayerModelObject* container, VisibleRectContext context) const
{
    if (document().settings().layerBasedSVGEngineEnabled()) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }
    return SVGRenderSupport::computeFloatVisibleRectInContainer(*this, rect, container, context);
}

void RenderSVGInline::mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState& transformState, OptionSet<MapCoordinatesMode> mode, bool* wasFixed) const
{
    if (document().settings().layerBasedSVGEngineEnabled()) {
        RenderInline::mapLocalToContainer(ancestorContainer, transformState, mode, wasFixed);
        return;
    }
    SVGRenderSupport::mapLocalToContainer(*this, ancestorContainer, transformState, wasFixed);
}

const RenderObject* RenderSVGInline::pushMappingToContainer(const RenderLayerModelObject* ancestorToStopAt, RenderGeometryMap& geometryMap) const
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return RenderInline::pushMappingToContainer(ancestorToStopAt, geometryMap);
    return SVGRenderSupport::pushMappingToContainer(*this, ancestorToStopAt, geometryMap);
}

void RenderSVGInline::absoluteQuads(Vector<FloatQuad>& quads, bool* wasFixed) const
{
    if (document().settings().layerBasedSVGEngineEnabled()) {
        RenderInline::absoluteQuads(quads, wasFixed);
        return;
    }

    auto* textAncestor = RenderSVGText::locateRenderSVGTextAncestor(*this);
    if (!textAncestor)
        return;

    FloatRect textBoundingBox = textAncestor->strokeBoundingBox();
    for (auto* box = firstLegacyInlineBox(); box; box = box->nextLineBox())
        quads.append(localToAbsoluteQuad(FloatRect(textBoundingBox.x() + box->x(), textBoundingBox.y() + box->y(), box->logicalWidth(), box->logicalHeight()), UseTransforms, wasFixed));
}

void RenderSVGInline::willBeDestroyed()
{
    if (document().settings().layerBasedSVGEngineEnabled()) {
        RenderInline::willBeDestroyed();
        return;
    }

    SVGResourcesCache::clientDestroyed(*this);
    RenderInline::willBeDestroyed();
}

void RenderSVGInline::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    if (document().settings().layerBasedSVGEngineEnabled()) {
        RenderInline::styleDidChange(diff, oldStyle);
        return;
    }

    if (diff == StyleDifference::Layout)
        invalidateCachedBoundaries();
    RenderInline::styleDidChange(diff, oldStyle);
    SVGResourcesCache::clientStyleChanged(*this, diff, oldStyle, style());
}

bool RenderSVGInline::needsHasSVGTransformFlags() const
{
    return graphicsElement().hasTransformRelatedAttributes();
}

void RenderSVGInline::updateFromStyle()
{
    RenderInline::updateFromStyle();

    if (document().settings().layerBasedSVGEngineEnabled())
        updateHasSVGTransformFlags();

    // SVG text layout code expects us to be an inline-level element.
    setInline(true);
}

}
