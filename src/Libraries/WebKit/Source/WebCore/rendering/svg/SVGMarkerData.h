/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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

#include "FloatConversion.h"
#include "Path.h"
#include <array>

namespace WebCore {

class RenderSVGResourceMarker;

enum SVGMarkerType {
    StartMarker,
    MidMarker,
    EndMarker
};

struct MarkerPosition {
    MarkerPosition(SVGMarkerType useType, const FloatPoint& useOrigin, float useAngle)
        : type(useType)
        , origin(useOrigin)
        , angle(useAngle)
    {
    }

    SVGMarkerType type;
    FloatPoint origin;
    float angle;
};

class SVGMarkerData {
public:
    SVGMarkerData(Vector<MarkerPosition>& positions, bool reverseStart)
        : m_positions(positions)
        , m_reverseStart(reverseStart)
    {
    }

    static void updateFromPathElement(SVGMarkerData& markerData, const PathElement& element)
    {
        // First update the outslope for the previous element.
        if (element.type != PathElement::Type::MoveToPoint)
            markerData.updateOutslope(element.points[0]);

        // Record the marker for the previous element.
        if (markerData.m_elementIndex > 0) {
            SVGMarkerType markerType = markerData.m_elementIndex == 1 ? StartMarker : MidMarker;
            SVGMarkerType markerTypeForOrientation;
            if (markerData.m_previousWasMoveTo)
                markerTypeForOrientation = StartMarker;
            else if (element.type == PathElement::Type::MoveToPoint)
                markerTypeForOrientation = EndMarker;
            else
                markerTypeForOrientation = markerType;
            markerData.m_positions.append(MarkerPosition(markerType, markerData.m_origin, markerData.currentAngle(markerTypeForOrientation)));
        }

        // Update our marker data for this element.
        markerData.updateMarkerDataForPathElement(element);
        markerData.m_previousWasMoveTo = element.type == PathElement::Type::MoveToPoint;
        ++markerData.m_elementIndex;
    }

    void pathIsDone()
    {
        m_positions.append(MarkerPosition(EndMarker, m_origin, currentAngle(EndMarker)));
    }

private:
    float currentAngle(SVGMarkerType type) const
    {
        // For details of this calculation, see: http://www.w3.org/TR/SVG/single-page.html#painting-MarkerElement
        FloatPoint inSlope(m_inslopePoints[1] - m_inslopePoints[0]);
        FloatPoint outSlope(m_outslopePoints[1] - m_outslopePoints[0]);

        double inAngle = rad2deg(inSlope.slopeAngleRadians());
        double outAngle = rad2deg(outSlope.slopeAngleRadians());

        switch (type) {
        case StartMarker:
            if (m_reverseStart)
                return narrowPrecisionToFloat(outAngle - 180);
            return narrowPrecisionToFloat(outAngle);
        case MidMarker:
            // WK193015: Prevent bugs due to angles being non-continuous.
            if (std::abs(inAngle - outAngle) > 180)
                inAngle += 360;
            return narrowPrecisionToFloat((inAngle + outAngle) / 2);
        case EndMarker:
            return narrowPrecisionToFloat(inAngle);
        }

        ASSERT_NOT_REACHED();
        return 0;
    }

    void updateOutslope(const FloatPoint& point)
    {
        m_outslopePoints[0] = m_origin;
        m_outslopePoints[1] = point;
    }

    void updateInslope(const FloatPoint& point)
    {
        m_inslopePoints[0] = m_origin;
        m_inslopePoints[1] = point;
    }

    void updateMarkerDataForPathElement(const PathElement& element)
    {
        auto& points = element.points;

        switch (element.type) {
        case PathElement::Type::AddQuadCurveToPoint:
            // FIXME: https://bugs.webkit.org/show_bug.cgi?id=33115 (PathElement::Type::AddQuadCurveToPoint not handled for <marker>)
            m_origin = points[1];
            break;
        case PathElement::Type::AddCurveToPoint:
            m_inslopePoints[0] = points[1];
            m_inslopePoints[1] = points[2];
            m_origin = points[2];
            break;
        case PathElement::Type::MoveToPoint:
            m_subpathStart = points[0];
            FALLTHROUGH;
        case PathElement::Type::AddLineToPoint:
            updateInslope(points[0]);
            m_origin = points[0];
            break;
        case PathElement::Type::CloseSubpath:
            updateInslope(points[0]);
            m_origin = m_subpathStart;
            m_subpathStart = FloatPoint();
        }
    }

    Vector<MarkerPosition>& m_positions;
    unsigned m_elementIndex { 0 };
    FloatPoint m_origin;
    FloatPoint m_subpathStart;
    std::array<FloatPoint, 2> m_inslopePoints;
    std::array<FloatPoint, 2> m_outslopePoints;
    bool m_reverseStart;
    bool m_previousWasMoveTo { false };
};

} // namespace WebCore
