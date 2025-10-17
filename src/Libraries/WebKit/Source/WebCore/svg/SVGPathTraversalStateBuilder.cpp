/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#include "SVGPathTraversalStateBuilder.h"

#include "PathTraversalState.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {

SVGPathTraversalStateBuilder::SVGPathTraversalStateBuilder(PathTraversalState& state, float desiredLength)
    : m_traversalState(state)
{
    m_traversalState.setDesiredLength(desiredLength);
}

void SVGPathTraversalStateBuilder::moveTo(const FloatPoint& targetPoint, bool, PathCoordinateMode)
{
    m_traversalState.processPathElement(PathElement::Type::MoveToPoint, singleElementSpan(targetPoint));
}

void SVGPathTraversalStateBuilder::lineTo(const FloatPoint& targetPoint, PathCoordinateMode)
{
    m_traversalState.processPathElement(PathElement::Type::AddLineToPoint, singleElementSpan(targetPoint));
}

void SVGPathTraversalStateBuilder::curveToCubic(const FloatPoint& point1, const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode)
{
    std::array points { point1, point2, targetPoint };

    m_traversalState.processPathElement(PathElement::Type::AddCurveToPoint, std::span<FloatPoint> { points });
}

void SVGPathTraversalStateBuilder::closePath()
{
    m_traversalState.processPathElement(PathElement::Type::CloseSubpath, { });
}

bool SVGPathTraversalStateBuilder::continueConsuming()
{
    return !m_traversalState.success();
}

float SVGPathTraversalStateBuilder::totalLength() const
{
    return m_traversalState.totalLength();
}

FloatPoint SVGPathTraversalStateBuilder::currentPoint() const
{
    return m_traversalState.current();
}

}
