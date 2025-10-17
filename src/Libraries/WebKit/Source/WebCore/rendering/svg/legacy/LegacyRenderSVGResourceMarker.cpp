/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#include "LegacyRenderSVGResourceMarker.h"

#include "GraphicsContext.h"
#include "LegacyRenderSVGResourceMarkerInlines.h"
#include "LegacyRenderSVGRoot.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGResourceMarker);

LegacyRenderSVGResourceMarker::LegacyRenderSVGResourceMarker(SVGMarkerElement& element, RenderStyle&& style)
    : LegacyRenderSVGResourceContainer(Type::LegacySVGResourceMarker, element, WTFMove(style))
{
}

LegacyRenderSVGResourceMarker::~LegacyRenderSVGResourceMarker() = default;

void LegacyRenderSVGResourceMarker::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    // Invalidate all resources if our layout changed.
    if (everHadLayout() && selfNeedsLayout())
        LegacyRenderSVGRoot::addResourceForClientInvalidation(this);

    // RenderSVGHiddenContainer overwrites layout(). We need the
    // layouting of LegacyRenderSVGContainer for calculating  local
    // transformations and repaint.
    LegacyRenderSVGContainer::layout();
}

void LegacyRenderSVGResourceMarker::applyViewportClip(PaintInfo& paintInfo)
{
    if (SVGRenderSupport::isOverflowHidden(*this))
        paintInfo.context().clip(m_viewport);
}

FloatRect LegacyRenderSVGResourceMarker::markerBoundaries(RepaintRectCalculation repaintRectCalculation, const AffineTransform& markerTransformation) const
{
    FloatRect coordinates = LegacyRenderSVGContainer::repaintRectInLocalCoordinates(repaintRectCalculation);

    // Map repaint rect into parent coordinate space, in which the marker boundaries have to be evaluated
    coordinates = localToParentTransform().mapRect(coordinates);

    return markerTransformation.mapRect(coordinates);
}

const AffineTransform& LegacyRenderSVGResourceMarker::localToParentTransform() const
{
    m_localToParentTransform = AffineTransform::makeTranslation(toFloatSize(m_viewport.location())) * viewportTransform();
    return m_localToParentTransform;
    // If this class were ever given a localTransform(), then the above would read:
    // return viewportTranslation * localTransform() * viewportTransform();
}

FloatPoint LegacyRenderSVGResourceMarker::referencePoint() const
{
    Ref markerElement = this->markerElement();
    SVGLengthContext lengthContext(markerElement.ptr());
    return FloatPoint(markerElement->refX().value(lengthContext), markerElement->refY().value(lengthContext));
}

std::optional<float> LegacyRenderSVGResourceMarker::angle() const
{
    if (markerElement().orientType() == SVGMarkerOrientAngle)
        return markerElement().orientAngle().value();
    return std::nullopt;
}

AffineTransform LegacyRenderSVGResourceMarker::markerTransformation(const FloatPoint& origin, float autoAngle, float strokeWidth) const
{
    bool useStrokeWidth = markerElement().markerUnits() == SVGMarkerUnitsStrokeWidth;

    AffineTransform transform;
    transform.translate(origin);
    transform.rotate(angle().value_or(autoAngle));
    transform = markerContentTransformation(transform, referencePoint(), useStrokeWidth ? strokeWidth : -1);
    return transform;
}

void LegacyRenderSVGResourceMarker::draw(PaintInfo& paintInfo, const AffineTransform& transform)
{
    // An empty viewBox disables rendering.
    if (markerElement().hasAttribute(SVGNames::viewBoxAttr) && markerElement().hasEmptyViewBox())
        return;

    PaintInfo info(paintInfo);
    GraphicsContextStateSaver stateSaver(info.context());
    info.applyTransform(transform);
    LegacyRenderSVGContainer::paint(info, IntPoint());
}

AffineTransform LegacyRenderSVGResourceMarker::markerContentTransformation(const AffineTransform& contentTransformation, const FloatPoint& origin, float strokeWidth) const
{
    // The 'origin' coordinate maps to SVGs refX/refY, given in coordinates relative to the viewport established by the marker
    FloatPoint mappedOrigin = viewportTransform().mapPoint(origin);

    AffineTransform transformation = contentTransformation;
    if (strokeWidth != -1)
        transformation.scaleNonUniform(strokeWidth, strokeWidth);

    transformation.translate(-mappedOrigin);
    return transformation;
}

AffineTransform LegacyRenderSVGResourceMarker::viewportTransform() const
{
    return markerElement().viewBoxToViewTransform(m_viewport.width(), m_viewport.height());
}

void LegacyRenderSVGResourceMarker::calcViewport()
{
    if (!selfNeedsLayout())
        return;

    Ref markerElement = this->markerElement();
    SVGLengthContext lengthContext(markerElement.ptr());
    float w = markerElement->markerWidth().value(lengthContext);
    float h = markerElement->markerHeight().value(lengthContext);
    m_viewport = FloatRect(0, 0, w, h);
}

}
