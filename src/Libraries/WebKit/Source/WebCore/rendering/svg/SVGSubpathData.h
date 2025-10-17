/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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

#include "Path.h"
#include <wtf/Forward.h>

namespace WebCore {

class SVGSubpathData {
public:
    SVGSubpathData(Vector<FloatPoint>& zeroLengthSubpathLocations)
        : m_zeroLengthSubpathLocations(zeroLengthSubpathLocations)
    {
    }

    static void updateFromPathElement(SVGSubpathData& subpathFinder, const PathElement& element)
    {
        switch (element.type) {
        case PathElement::Type::MoveToPoint:
            if (subpathFinder.m_pathIsZeroLength && !subpathFinder.m_haveSeenMoveOnly)
                subpathFinder.m_zeroLengthSubpathLocations.append(subpathFinder.m_lastPoint);
            subpathFinder.m_lastPoint = subpathFinder.m_movePoint = element.points[0];
            subpathFinder.m_haveSeenMoveOnly = true;
            subpathFinder.m_pathIsZeroLength = true;
            break;
        case PathElement::Type::AddLineToPoint:
            if (subpathFinder.m_lastPoint != element.points[0]) {
                subpathFinder.m_pathIsZeroLength = false;
                subpathFinder.m_lastPoint = element.points[0];
            }
            subpathFinder.m_haveSeenMoveOnly = false;
            break;
        case PathElement::Type::AddQuadCurveToPoint:
            if (subpathFinder.m_lastPoint != element.points[0] || element.points[0] != element.points[1]) {
                subpathFinder.m_pathIsZeroLength = false;
                subpathFinder.m_lastPoint = element.points[1];
            }
            subpathFinder.m_haveSeenMoveOnly = false;
            break;
        case PathElement::Type::AddCurveToPoint:
            if (subpathFinder.m_lastPoint != element.points[0] || element.points[0] != element.points[1] || element.points[1] != element.points[2]) {
                subpathFinder.m_pathIsZeroLength = false;
                subpathFinder.m_lastPoint = element.points[2];
            }
            subpathFinder.m_haveSeenMoveOnly = false;
            break;
        case PathElement::Type::CloseSubpath:
            if (subpathFinder.m_pathIsZeroLength)
                subpathFinder.m_zeroLengthSubpathLocations.append(subpathFinder.m_lastPoint);
            subpathFinder.m_haveSeenMoveOnly = true; // This is an implicit move for the next element
            subpathFinder.m_pathIsZeroLength = true; // A new sub-path also starts here
            subpathFinder.m_lastPoint = subpathFinder.m_movePoint;
            break;
        }
    }

    void pathIsDone()
    {
        if (m_pathIsZeroLength && !m_haveSeenMoveOnly)
            m_zeroLengthSubpathLocations.append(m_lastPoint);
    }

private:
    Vector<FloatPoint>& m_zeroLengthSubpathLocations;
    FloatPoint m_lastPoint;
    FloatPoint m_movePoint;
    bool m_haveSeenMoveOnly { false };
    bool m_pathIsZeroLength { false };
};

} // namespace WebCore
