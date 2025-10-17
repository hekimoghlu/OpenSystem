/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include "RenderSVGViewportContainer.h"

#include "RenderLayer.h"
#include "RenderSVGModelObjectInlines.h"
#include "RenderSVGRoot.h"
#include "SVGContainerLayout.h"
#include "SVGElementTypeHelpers.h"
#include "SVGSVGElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGViewportContainer);

RenderSVGViewportContainer::RenderSVGViewportContainer(RenderSVGRoot& parent, RenderStyle&& style)
    : RenderSVGContainer(Type::SVGViewportContainer, parent.document(), WTFMove(style))
    , m_owningSVGRoot(parent)
{
    ASSERT(isRenderSVGViewportContainer());
}

RenderSVGViewportContainer::RenderSVGViewportContainer(SVGSVGElement& element, RenderStyle&& style)
    : RenderSVGContainer(Type::SVGViewportContainer, element, WTFMove(style))
{
    ASSERT(isRenderSVGViewportContainer());
}

RenderSVGViewportContainer::~RenderSVGViewportContainer() = default;

SVGSVGElement& RenderSVGViewportContainer::svgSVGElement() const
{
    if (isOutermostSVGViewportContainer()) {
        ASSERT(m_owningSVGRoot);
        return m_owningSVGRoot->svgSVGElement();
    }
    return downcast<SVGSVGElement>(RenderSVGContainer::element());
}

Ref<SVGSVGElement> RenderSVGViewportContainer::protectedSVGSVGElement() const
{
    return svgSVGElement();
}

FloatPoint RenderSVGViewportContainer::computeViewportLocation() const
{
    if (isOutermostSVGViewportContainer())
        return { };

    Ref useSVGSVGElement = svgSVGElement();
    SVGLengthContext lengthContext(useSVGSVGElement.ptr());
    return { useSVGSVGElement->x().value(lengthContext), useSVGSVGElement->y().value(lengthContext) };
}

FloatSize RenderSVGViewportContainer::computeViewportSize() const
{
    if (isOutermostSVGViewportContainer())
        return downcast<RenderSVGRoot>(*parent()).computeViewportSize();

    Ref useSVGSVGElement = svgSVGElement();
    SVGLengthContext lengthContext(useSVGSVGElement.ptr());
    return { useSVGSVGElement->width().value(lengthContext), useSVGSVGElement->height().value(lengthContext) };
}

bool RenderSVGViewportContainer::updateLayoutSizeIfNeeded()
{
    auto previousViewportSize = viewportSize();
    m_viewport = { computeViewportLocation(), computeViewportSize() };
    return selfNeedsLayout() || previousViewportSize != viewportSize();
}

bool RenderSVGViewportContainer::needsHasSVGTransformFlags() const
{
    Ref useSVGSVGElement = svgSVGElement();
    if (useSVGSVGElement->hasTransformRelatedAttributes())
        return true;

    if (isOutermostSVGViewportContainer())
        return !useSVGSVGElement->currentTranslateValue().isZero() || useSVGSVGElement->renderer()->style().usedZoom() != 1;

    return false;
}

void RenderSVGViewportContainer::updateFromStyle()
{
    RenderSVGContainer::updateFromStyle();

    if (SVGRenderSupport::isOverflowHidden(*this))
        setHasNonVisibleOverflow();
}

inline AffineTransform viewBoxToViewTransform(const SVGSVGElement& svgSVGElement, const FloatSize& viewportSize)
{
    return svgSVGElement.viewBoxToViewTransform(viewportSize.width(), viewportSize.height());
}

void RenderSVGViewportContainer::updateLayerTransform()
{
    ASSERT(hasLayer());

    // First update the supplemental layer transform.
    Ref useSVGSVGElement = svgSVGElement();
    auto viewportSize = this->viewportSize();

    m_supplementalLayerTransform.makeIdentity();

    if (isOutermostSVGViewportContainer()) {
        // Handle pan - set on outermost <svg> element.
        if (auto translation = useSVGSVGElement->currentTranslateValue(); !translation.isZero())
            m_supplementalLayerTransform.translate(translation);

        // Handle zoom - take effective zoom from outermost <svg> element.
        if (auto scale = useSVGSVGElement->renderer()->style().usedZoom(); scale != 1) {
            m_supplementalLayerTransform.scale(scale);
            viewportSize.scale(1.0 / scale);
        }
    } else if (!m_viewport.location().isZero())
        m_supplementalLayerTransform.translate(m_viewport.location());

    if (useSVGSVGElement->hasAttribute(SVGNames::viewBoxAttr)) {
        // An empty viewBox disables the rendering -- dirty the visible descendant status!
        if (useSVGSVGElement->hasEmptyViewBox())
            layer()->dirtyVisibleContentStatus();
        else if (auto viewBoxTransform = viewBoxToViewTransform(useSVGSVGElement, viewportSize); !viewBoxTransform.isIdentity()) {
            if (m_supplementalLayerTransform.isIdentity())
                m_supplementalLayerTransform = viewBoxTransform;
            else
                m_supplementalLayerTransform.multiply(viewBoxTransform);
        }
    }

    // After updating the supplemental layer transform we're able to use it in RenderLayerModelObjects::updateLayerTransform().
    RenderSVGContainer::updateLayerTransform();
}

void RenderSVGViewportContainer::applyTransform(TransformationMatrix& transform, const RenderStyle& style, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption> options) const
{
    applySVGTransform(transform, protectedSVGSVGElement(), style, boundingBox, m_supplementalLayerTransform.isIdentity() ? std::nullopt : std::make_optional(m_supplementalLayerTransform), std::nullopt, options);
}

LayoutRect RenderSVGViewportContainer::overflowClipRect(const LayoutPoint& location, OverlayScrollbarSizeRelevancy, PaintPhase) const
{
    // Overflow for the outermost <svg> element is handled in RenderSVGRoot, not here.
    ASSERT(!isOutermostSVGViewportContainer());
    Ref useSVGSVGElement = svgSVGElement();

    auto clipRect = enclosingLayoutRect(viewport());
    if (useSVGSVGElement->hasAttribute(SVGNames::viewBoxAttr)) {
        if (useSVGSVGElement->hasEmptyViewBox())
            return { };

        if (auto viewBoxTransform = viewBoxToViewTransform(useSVGSVGElement, viewportSize()); !viewBoxTransform.isIdentity())
            clipRect = enclosingLayoutRect(viewBoxTransform.inverse().value_or(AffineTransform { }).mapRect(viewport()));
    }

    clipRect.moveBy(location);
    return clipRect;
}

}

