/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#include "LegacyRenderSVGPath.h"

#include "Gradient.h"
#include "LegacyRenderSVGShapeInlines.h"
#include "RenderStyleInlines.h"
#include "SVGElementTypeHelpers.h"
#include "SVGPathElement.h"
#include "SVGRenderStyle.h"
#include "SVGResources.h"
#include "SVGResourcesCache.h"
#include "SVGSubpathData.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGPath);

LegacyRenderSVGPath::LegacyRenderSVGPath(SVGGraphicsElement& element, RenderStyle&& style)
    : LegacyRenderSVGShape(Type::LegacySVGPath, element, WTFMove(style))
{
    ASSERT(isLegacyRenderSVGPath());
}

LegacyRenderSVGPath::~LegacyRenderSVGPath() = default;

void LegacyRenderSVGPath::updateShapeFromElement()
{
    clearPath();
    m_shapeType = ShapeType::Empty;
    m_fillBoundingBox = ensurePath().boundingRect();
    m_strokeBoundingBox = std::nullopt;
    m_approximateStrokeBoundingBox = std::nullopt;
    processMarkerPositions();
    updateZeroLengthSubpaths();

    ASSERT(hasPath());
    if (path().isEmpty())
        return;
    if (path().definitelySingleLine())
        m_shapeType = ShapeType::Line;
    else
        m_shapeType = ShapeType::Path;

    // FIXME: This should not exist. However, currently SVG is relying on ordering of calculation of SVG2 strokeBoundingBox via layout() function
    // for recursive SVGs (markers are pointing to each other recursively). If we move to SVG2 computation, we no longer need this since SVG2 strokeBoundingBox
    // does not include markers rect (so we do not need to have this in LSBE). Right now, this exists only for RepaintRectCalculation::Accurate, and it should be removed once
    // 1. We fix checkInsertion / checkEnclosure implementations. Currently they are not aligned to what the spec requires.
    // 2. We move our RenderTreeAsText to avoid dumping Accurate repaint rect. We should dump strokeBoundingBox or actual repaintBoundingBox instead.
    // We fall back to path-based eager strokeBoundingBox only when there are markers and it is not SVG2.
    // There are several cases we use approximate repaintBoundingBox. But only LegacyRenderSVGPath can reference to the other approximate repaintBoundingBox via markers.
    // The other resources including maskers, clippers etc. are already computing bounding rect via Accurate eagerly. So they do not matter.
    // https://bugs.webkit.org/show_bug.cgi?id=263348
    if (!m_markerPositions.isEmpty())
        strokeBoundingBox();
}

FloatRect LegacyRenderSVGPath::adjustStrokeBoundingBoxForMarkersAndZeroLengthLinecaps(RepaintRectCalculation repaintRectCalculation, FloatRect strokeBoundingBox) const
{
    float strokeWidth = this->strokeWidth();

    if (!m_markerPositions.isEmpty())
        strokeBoundingBox.unite(markerRect(repaintRectCalculation, strokeWidth));

    if (style().svgStyle().hasStroke()) {
        // FIXME: zero-length subpaths do not respect vector-effect = non-scaling-stroke.
        for (size_t i = 0; i < m_zeroLengthLinecapLocations.size(); ++i)
            strokeBoundingBox.unite(zeroLengthSubpathRect(m_zeroLengthLinecapLocations[i], strokeWidth));
    }

    return strokeBoundingBox;
}

static void useStrokeStyleToFill(GraphicsContext& context)
{
    if (auto gradient = context.strokeGradient())
        context.setFillGradient(*gradient, context.strokeGradientSpaceTransform());
    else if (Pattern* pattern = context.strokePattern())
        context.setFillPattern(*pattern);
    else
        context.setFillColor(context.strokeColor());
}

void LegacyRenderSVGPath::strokeShape(GraphicsContext& context) const
{
    if (!style().hasVisibleStroke())
        return;

    // This happens only if the layout was never been called for this element.
    if (!hasPath())
        return;

    LegacyRenderSVGShape::strokeShape(context);
    strokeZeroLengthSubpaths(context);
}

bool LegacyRenderSVGPath::shapeDependentStrokeContains(const FloatPoint& point, PointCoordinateSpace pointCoordinateSpace)
{
    if (LegacyRenderSVGShape::shapeDependentStrokeContains(point, pointCoordinateSpace))
        return true;

    for (size_t i = 0; i < m_zeroLengthLinecapLocations.size(); ++i) {
        ASSERT(style().svgStyle().hasStroke());
        float strokeWidth = this->strokeWidth();
        if (style().capStyle() == LineCap::Square) {
            if (zeroLengthSubpathRect(m_zeroLengthLinecapLocations[i], strokeWidth).contains(point))
                return true;
        } else {
            ASSERT(style().capStyle() == LineCap::Round);
            FloatPoint radiusVector(point.x() - m_zeroLengthLinecapLocations[i].x(), point.y() -  m_zeroLengthLinecapLocations[i].y());
            if (radiusVector.lengthSquared() < strokeWidth * strokeWidth * .25f)
                return true;
        }
    }
    return false;
}

bool LegacyRenderSVGPath::shouldStrokeZeroLengthSubpath() const
{
    // Spec(11.4): Any zero length subpath shall not be stroked if the "stroke-linecap" property has a value of butt
    // but shall be stroked if the "stroke-linecap" property has a value of round or square
    return style().svgStyle().hasStroke() && style().capStyle() != LineCap::Butt;
}

Path* LegacyRenderSVGPath::zeroLengthLinecapPath(const FloatPoint& linecapPosition) const
{
    static NeverDestroyed<Path> tempPath;

    tempPath.get().clear();
    if (style().capStyle() == LineCap::Square)
        tempPath.get().addRect(zeroLengthSubpathRect(linecapPosition, this->strokeWidth()));
    else
        tempPath.get().addEllipseInRect(zeroLengthSubpathRect(linecapPosition, this->strokeWidth()));

    return &tempPath.get();
}

FloatRect LegacyRenderSVGPath::zeroLengthSubpathRect(const FloatPoint& linecapPosition, float strokeWidth) const
{
    return FloatRect(linecapPosition.x() - strokeWidth / 2, linecapPosition.y() - strokeWidth / 2, strokeWidth, strokeWidth);
}

void LegacyRenderSVGPath::updateZeroLengthSubpaths()
{
    m_zeroLengthLinecapLocations.clear();

    if (!strokeWidth() || !shouldStrokeZeroLengthSubpath())
        return;

    SVGSubpathData subpathData(m_zeroLengthLinecapLocations);
    path().applyElements([&subpathData](const PathElement& pathElement) {
        SVGSubpathData::updateFromPathElement(subpathData, pathElement);
    });
    subpathData.pathIsDone();
}

void LegacyRenderSVGPath::strokeZeroLengthSubpaths(GraphicsContext& context) const
{
    if (m_zeroLengthLinecapLocations.isEmpty())
        return;

    AffineTransform nonScalingTransform;
    if (hasNonScalingStroke())
        nonScalingTransform = nonScalingStrokeTransform();

    GraphicsContextStateSaver stateSaver(context, true);
    useStrokeStyleToFill(context);
    for (size_t i = 0; i < m_zeroLengthLinecapLocations.size(); ++i) {
        auto usePath = zeroLengthLinecapPath(m_zeroLengthLinecapLocations[i]);
        if (hasNonScalingStroke())
            usePath = nonScalingStrokePath(usePath, nonScalingTransform);
        context.fillPath(*usePath);
    }
}

static inline LegacyRenderSVGResourceMarker* markerForType(SVGMarkerType type, LegacyRenderSVGResourceMarker* markerStart, LegacyRenderSVGResourceMarker* markerMid, LegacyRenderSVGResourceMarker* markerEnd)
{
    switch (type) {
    case StartMarker:
        return markerStart;
    case MidMarker:
        return markerMid;
    case EndMarker:
        return markerEnd;
    }

    ASSERT_NOT_REACHED();
    return 0;
}

bool LegacyRenderSVGPath::shouldGenerateMarkerPositions() const
{
    if (!style().svgStyle().hasMarkers())
        return false;

    if (!graphicsElement().supportsMarkers())
        return false;

    auto* resources = SVGResourcesCache::cachedResourcesForRenderer(*this);
    if (!resources)
        return false;

    return resources->markerStart() || resources->markerMid() || resources->markerEnd();
}

void LegacyRenderSVGPath::drawMarkers(PaintInfo& paintInfo)
{
    if (m_markerPositions.isEmpty())
        return;

    auto* resources = SVGResourcesCache::cachedResourcesForRenderer(*this);
    if (!resources)
        return;

    LegacyRenderSVGResourceMarker* markerStart = resources->markerStart();
    LegacyRenderSVGResourceMarker* markerMid = resources->markerMid();
    LegacyRenderSVGResourceMarker* markerEnd = resources->markerEnd();
    if (!markerStart && !markerMid && !markerEnd)
        return;

    float strokeWidth = this->strokeWidth();
    unsigned size = m_markerPositions.size();
    for (unsigned i = 0; i < size; ++i) {
        if (auto* marker = markerForType(m_markerPositions[i].type, markerStart, markerMid, markerEnd))
            marker->draw(paintInfo, marker->markerTransformation(m_markerPositions[i].origin, m_markerPositions[i].angle, strokeWidth));
    }
}

FloatRect LegacyRenderSVGPath::markerRect(RepaintRectCalculation repaintRectCalculation, float strokeWidth) const
{
    ASSERT(!m_markerPositions.isEmpty());

    auto* resources = SVGResourcesCache::cachedResourcesForRenderer(*this);
    ASSERT(resources);

    auto* markerStart = resources->markerStart();
    auto* markerMid = resources->markerMid();
    auto* markerEnd = resources->markerEnd();
    ASSERT(markerStart || markerMid || markerEnd);

    FloatRect boundaries;
    unsigned size = m_markerPositions.size();
    for (unsigned i = 0; i < size; ++i) {
        if (auto* marker = markerForType(m_markerPositions[i].type, markerStart, markerMid, markerEnd))
            boundaries.unite(marker->markerBoundaries(repaintRectCalculation, marker->markerTransformation(m_markerPositions[i].origin, m_markerPositions[i].angle, strokeWidth)));
    }
    return boundaries;
}

void LegacyRenderSVGPath::processMarkerPositions()
{
    m_markerPositions.clear();

    if (!shouldGenerateMarkerPositions())
        return;

    ASSERT(hasPath());

    SVGMarkerData markerData(m_markerPositions, SVGResourcesCache::cachedResourcesForRenderer(*this)->markerReverseStart());
    path().applyElements([&markerData](const PathElement& pathElement) {
        SVGMarkerData::updateFromPathElement(markerData, pathElement);
    });
    markerData.pathIsDone();
}

bool LegacyRenderSVGPath::isRenderingDisabled() const
{
    // For a polygon, polyline or path, rendering is disabled if there is no path data.
    // No path data is possible in the case of a missing or empty 'd' or 'points' attribute.
    return !hasPath() || path().isEmpty();
}

void LegacyRenderSVGPath::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    if (RefPtr pathElement = dynamicDowncast<SVGPathElement>(graphicsElement())) {
        if (!oldStyle || style().d() != oldStyle->d())
            pathElement->pathDidChange();
    }

    LegacyRenderSVGShape::styleDidChange(diff, oldStyle);
}

}
