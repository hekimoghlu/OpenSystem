/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#include "PathTraversalState.h"

#include "GeometryUtilities.h"
#include <wtf/MathExtras.h>
#include <wtf/Vector.h>

namespace WebCore {

static const float kPathSegmentLengthTolerance = 0.00001f;

static inline float distanceLine(const FloatPoint& start, const FloatPoint& end)
{
    return std::hypot(end.x() - start.x(), end.y() - start.y());
}

struct QuadraticBezier {
    QuadraticBezier() = default;
    QuadraticBezier(const FloatPoint& s, const FloatPoint& c, const FloatPoint& e)
        : start(s)
        , control(c)
        , end(e)
    {
    }

    friend bool operator==(const QuadraticBezier&, const QuadraticBezier&) = default;
    
    float approximateDistance() const
    {
        return distanceLine(start, control) + distanceLine(control, end);
    }
    
    bool split(QuadraticBezier& left, QuadraticBezier& right) const
    {
        left.control = midPoint(start, control);
        right.control = midPoint(control, end);
        
        FloatPoint leftControlToRightControl = midPoint(left.control, right.control);
        left.end = leftControlToRightControl;
        right.start = leftControlToRightControl;

        left.start = start;
        right.end = end;

        return !(left == *this) && !(right == *this);
    }
    
    FloatPoint start;
    FloatPoint control;
    FloatPoint end;
};

struct CubicBezier {
    CubicBezier() = default;
    CubicBezier(const FloatPoint& s, const FloatPoint& c1, const FloatPoint& c2, const FloatPoint& e)
        : start(s)
        , control1(c1)
        , control2(c2)
        , end(e)
    {
    }

    friend bool operator==(const CubicBezier&, const CubicBezier&) = default;

    float approximateDistance() const
    {
        return distanceLine(start, control1) + distanceLine(control1, control2) + distanceLine(control2, end);
    }
        
    bool split(CubicBezier& left, CubicBezier& right) const
    {    
        FloatPoint startToControl1 = midPoint(control1, control2);
        
        left.start = start;
        left.control1 = midPoint(start, control1);
        left.control2 = midPoint(left.control1, startToControl1);
        
        right.control2 = midPoint(control2, end);
        right.control1 = midPoint(right.control2, startToControl1);
        right.end = end;
        
        FloatPoint leftControl2ToRightControl1 = midPoint(left.control2, right.control1);
        left.end = leftControl2ToRightControl1;
        right.start = leftControl2ToRightControl1;

        return !(left == *this) && !(right == *this);
    }
    
    FloatPoint start;
    FloatPoint control1;
    FloatPoint control2;
    FloatPoint end;
};

// FIXME: This function is possibly very slow due to the ifs required for proper path measuring
// A simple speed-up would be to use an additional boolean template parameter to control whether
// to use the "fast" version of this function with no PathTraversalState updating, vs. the slow
// version which does update the PathTraversalState.  We'll have to shark it to see if that's necessary.
// Another check which is possible up-front (to send us down the fast path) would be to check if
// approximateDistance() + current total distance > desired distance
template<class CurveType>
static float curveLength(const PathTraversalState& traversalState, const CurveType& originalCurve, FloatPoint& previous, FloatPoint& current)
{
    static const unsigned curveStackDepthLimit = 20;
    CurveType curve = originalCurve;
    Vector<CurveType, curveStackDepthLimit> curveStack;
    float totalLength = 0;

    while (true) {
        float length = curve.approximateDistance();

        CurveType leftCurve;
        CurveType rightCurve;

        if ((length - distanceLine(curve.start, curve.end)) > kPathSegmentLengthTolerance && curveStack.size() < curveStackDepthLimit && curve.split(leftCurve, rightCurve)) {
            curve = leftCurve;
            curveStack.append(rightCurve);
            continue;
        }

        totalLength += length;
        if (traversalState.action() == PathTraversalState::Action::VectorAtLength) {
            previous = curve.start;
            current = curve.end;
            if (traversalState.totalLength() + totalLength > traversalState.desiredLength())
                break;
        }

        if (curveStack.isEmpty())
            break;

        curve = curveStack.last();
        curveStack.removeLast();
    }

    if (traversalState.action() != PathTraversalState::Action::VectorAtLength) {
        ASSERT(curve.end == originalCurve.end);
        previous = curve.start;
        current = curve.end;
    }

    return totalLength;
}

PathTraversalState::PathTraversalState(Action action, float desiredLength)
    : m_action(action)
    , m_desiredLength(desiredLength)
{
    ASSERT(action != Action::TotalLength || !desiredLength);
}

void PathTraversalState::closeSubpath()
{
    m_totalLength += distanceLine(m_current, m_start);
    m_current = m_start;
}

void PathTraversalState::moveTo(const FloatPoint& point)
{
    m_previous = m_current = m_start = point;
}

void PathTraversalState::lineTo(const FloatPoint& point)
{
    m_totalLength += distanceLine(m_current, point);
    m_current = point;
}

void PathTraversalState::quadraticBezierTo(const FloatPoint& newControl, const FloatPoint& newEnd)
{
    m_totalLength += curveLength<QuadraticBezier>(*this, QuadraticBezier(m_current, newControl, newEnd), m_previous, m_current);
}

void PathTraversalState::cubicBezierTo(const FloatPoint& newControl1, const FloatPoint& newControl2, const FloatPoint& newEnd)
{
    m_totalLength += curveLength<CubicBezier>(*this, CubicBezier(m_current, newControl1, newControl2, newEnd), m_previous, m_current);
}

bool PathTraversalState::finalizeAppendPathElement()
{
    if (m_action == Action::TotalLength)
        return false;

    if (m_action == Action::SegmentAtLength) {
        if (m_totalLength >= m_desiredLength)
            m_success = true;
        return m_success;
    }

    ASSERT(m_action == Action::VectorAtLength);

    if (m_totalLength >= m_desiredLength) {
        float slope = FloatPoint(m_current - m_previous).slopeAngleRadians();
        float offset = m_desiredLength - m_totalLength;
        m_current.move(offset * cosf(slope), offset * sinf(slope));

        if (!m_isZeroVector && !m_desiredLength)
            m_isZeroVector = true;
        else {
            m_success = true;
            m_normalAngle = rad2deg(slope);
        }
    }

    m_previous = m_current;
    return m_success;
}

bool PathTraversalState::appendPathElement(PathElement::Type type, std::span<const FloatPoint> points)
{
    switch (type) {
    case PathElement::Type::MoveToPoint:
        moveTo(points[0]);
        break;
    case PathElement::Type::AddLineToPoint:
        lineTo(points[0]);
        break;
    case PathElement::Type::AddQuadCurveToPoint:
        quadraticBezierTo(points[0], points[1]);
        break;
    case PathElement::Type::AddCurveToPoint:
        cubicBezierTo(points[0], points[1], points[2]);
        break;
    case PathElement::Type::CloseSubpath:
        closeSubpath();
        break;
    }
    
    return finalizeAppendPathElement();
}

bool PathTraversalState::processPathElement(PathElement::Type type, std::span<const FloatPoint> points)
{
    if (m_success)
        return true;

    if (m_isZeroVector) {
        PathTraversalState traversalState(*this);
        m_success = traversalState.appendPathElement(type, points);
        m_normalAngle = traversalState.m_normalAngle;
        return m_success;
    }

    return appendPathElement(type, points);
}

}
