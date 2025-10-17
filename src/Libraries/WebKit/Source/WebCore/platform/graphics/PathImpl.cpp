/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#include "PathImpl.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PathImpl);

void PathImpl::addLinesForRect(const FloatRect& rect)
{
    add(PathMoveTo { rect.minXMinYCorner() });
    add(PathLineTo { rect.maxXMinYCorner() });
    add(PathLineTo { rect.maxXMaxYCorner() });
    add(PathLineTo { rect.minXMaxYCorner() });
    add(PathCloseSubpath { });
}

void PathImpl::addBeziersForRoundedRect(const FloatRoundedRect& roundedRect)
{
    const auto& radii = roundedRect.radii();
    const auto& rect = roundedRect.rect();

    const auto& topLeftRadius = radii.topLeft();
    const auto& topRightRadius = radii.topRight();
    const auto& bottomLeftRadius = radii.bottomLeft();
    const auto& bottomRightRadius = radii.bottomRight();

    add(PathMoveTo { FloatPoint(rect.x() + topLeftRadius.width(), rect.y()) });

    add(PathLineTo { FloatPoint(rect.maxX() - topRightRadius.width(), rect.y()) });
    if (topRightRadius.width() > 0 || topRightRadius.height() > 0) {
        add(PathBezierCurveTo { FloatPoint(rect.maxX() - topRightRadius.width() * circleControlPoint(), rect.y()),
            FloatPoint(rect.maxX(), rect.y() + topRightRadius.height() * circleControlPoint()),
            FloatPoint(rect.maxX(), rect.y() + topRightRadius.height()) });
    }

    add(PathLineTo { FloatPoint(rect.maxX(), rect.maxY() - bottomRightRadius.height()) });
    if (bottomRightRadius.width() > 0 || bottomRightRadius.height() > 0) {
        add(PathBezierCurveTo { FloatPoint(rect.maxX(), rect.maxY() - bottomRightRadius.height() * circleControlPoint()),
            FloatPoint(rect.maxX() - bottomRightRadius.width() * circleControlPoint(), rect.maxY()),
            FloatPoint(rect.maxX() - bottomRightRadius.width(), rect.maxY()) });
    }

    add(PathLineTo { FloatPoint(rect.x() + bottomLeftRadius.width(), rect.maxY()) });
    if (bottomLeftRadius.width() > 0 || bottomLeftRadius.height() > 0) {
        add(PathBezierCurveTo { FloatPoint(rect.x() + bottomLeftRadius.width() * circleControlPoint(), rect.maxY()),
            FloatPoint(rect.x(), rect.maxY() - bottomLeftRadius.height() * circleControlPoint()),
            FloatPoint(rect.x(), rect.maxY() - bottomLeftRadius.height()) });
    }

    add(PathLineTo { FloatPoint(rect.x(), rect.y() + topLeftRadius.height()) });
    if (topLeftRadius.width() > 0 || topLeftRadius.height() > 0) {
        add(PathBezierCurveTo { FloatPoint(rect.x(), rect.y() + topLeftRadius.height() * circleControlPoint()),
            FloatPoint(rect.x() + topLeftRadius.width() * circleControlPoint(), rect.y()),
            FloatPoint(rect.x() + topLeftRadius.width(), rect.y()) });
    }

    add(PathCloseSubpath { });
}

void PathImpl::applySegments(const PathSegmentApplier& applier) const
{
    applyElements([&](const PathElement& pathElement) {
        switch (pathElement.type) {
        case PathElement::Type::MoveToPoint:
            applier({ PathMoveTo { pathElement.points[0] } });
            break;

        case PathElement::Type::AddLineToPoint:
            applier({ PathLineTo { pathElement.points[0] } });
            break;

        case PathElement::Type::AddQuadCurveToPoint:
            applier({ PathQuadCurveTo { pathElement.points[0], pathElement.points[1] } });
            break;

        case PathElement::Type::AddCurveToPoint:
            applier({ PathBezierCurveTo { pathElement.points[0], pathElement.points[1], pathElement.points[2] } });
            break;

        case PathElement::Type::CloseSubpath:
            applier({ PathCloseSubpath { } });
            break;
        }
    });
}

bool PathImpl::isClosed() const
{
    bool lastElementIsClosed = false;

    // The path is closed if the type of the last PathElement is CloseSubpath. Unfortunately,
    // the only way to access PathElements is sequentially through apply(), there's no random
    // access as if they're in a vector.
    // The lambda below sets lastElementIsClosed if the last PathElement is CloseSubpath.
    // Because lastElementIsClosed is overridden if there are any remaining PathElements
    // to be iterated, its final value is the value of the last iteration.
    // (i.e the last PathElement).
    // FIXME: find a more efficient way to implement this, that does not require iterating
    // through all PathElements.
    applyElements([&lastElementIsClosed](const PathElement& element) {
        lastElementIsClosed = (element.type == PathElement::Type::CloseSubpath);
    });

    return lastElementIsClosed;
}

bool PathImpl::hasSubpaths() const
{
    auto rect = fastBoundingRect();
    return rect.height() || rect.width();
}

} // namespace WebCore
