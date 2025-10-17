/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#include "StyleEllipseFunction.h"

#include "FloatRect.h"
#include "GeometryUtilities.h"
#include "Path.h"
#include "StyleGradient.h"
#include "StylePrimitiveNumericTypes+Blending.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"
#include <wtf/TinyLRUCache.h>

namespace WebCore {
namespace Style {

// MARK: - Path Caching

struct EllipsePathPolicy final : public TinyLRUCachePolicy<FloatRect, WebCore::Path> {
public:
    static bool isKeyNull(const FloatRect& rect)
    {
        return rect.isEmpty();
    }

    static WebCore::Path createValueForKey(const FloatRect& rect)
    {
        WebCore::Path path;
        path.addEllipseInRect(rect);
        return path;
    }
};

static const WebCore::Path& cachedEllipsePath(const FloatRect& rect)
{
    static NeverDestroyed<TinyLRUCache<FloatRect, WebCore::Path, 4, EllipsePathPolicy>> cache;
    return cache.get().get(rect);
}

// MARK: - Path Generation

FloatPoint resolvePosition(const Ellipse& value, FloatSize boundingBox)
{
    return value.position ? evaluate(*value.position, boundingBox) : FloatPoint { boundingBox.width() / 2, boundingBox.height() / 2 };
}

FloatSize resolveRadii(const Ellipse& value, FloatSize boxSize, FloatPoint center)
{
    auto sizeForAxis = [&](const Ellipse::RadialSize& radius, float centerValue, float dimensionSize) {
        return WTF::switchOn(radius,
            [&](const Ellipse::Length& length) -> float {
                return evaluate(length, std::abs(dimensionSize));
            },
            [&](const Ellipse::Extent& extent) -> float {
                return WTF::switchOn(extent,
                    [&](CSS::Keyword::ClosestSide) -> float {
                        return std::min(std::abs(centerValue), std::abs(dimensionSize - centerValue));
                    },
                    [&](CSS::Keyword::FarthestSide) -> float {
                        return std::max(std::abs(centerValue), std::abs(dimensionSize - centerValue));
                    },
                    [&](CSS::Keyword::ClosestCorner) -> float {
                        return distanceToClosestCorner(center, boxSize);
                    },
                    [&](CSS::Keyword::FarthestCorner) -> float {
                        return distanceToFarthestCorner(center, boxSize);
                    }
                );
            }
        );
    };

    return {
        sizeForAxis(get<0>(value.radii), center.x(), boxSize.width()),
        sizeForAxis(get<1>(value.radii), center.y(), boxSize.height())
    };
}

WebCore::Path pathForCenterCoordinate(const Ellipse& value, const FloatRect& boundingBox, FloatPoint center)
{
    auto radii = resolveRadii(value, boundingBox.size(), center);
    auto bounding = FloatRect {
        center.x() - radii.width() + boundingBox.x(),
        center.y() - radii.height() + boundingBox.y(),
        radii.width() * 2,
        radii.height() * 2
    };
    return cachedEllipsePath(bounding);
}

WebCore::Path PathComputation<Ellipse>::operator()(const Ellipse& value, const FloatRect& boundingBox)
{
    return pathForCenterCoordinate(value, boundingBox, resolvePosition(value, boundingBox.size()));
}

// MARK: - Blending

auto Blending<Ellipse>::canBlend(const Ellipse& a, const Ellipse& b) -> bool
{
    auto canBlendRadius = [](const auto& radiusA, const auto& radiusB) {
        return std::visit(WTF::makeVisitor(
            [](const Ellipse::Length&, const Ellipse::Length&) {
                return true;
            },
            [](const auto&, const auto&) {
                // FIXME: Determine how to interpolate between keywords. See bug 125108.
                return false;
            }
        ), radiusA, radiusB);
    };

    return canBlendRadius(get<0>(a.radii), get<0>(b.radii))
        && canBlendRadius(get<1>(a.radii), get<1>(b.radii))
        && WebCore::Style::canBlend(a.position, b.position);
}

auto Blending<Ellipse>::blend(const Ellipse& a, const Ellipse& b, const BlendingContext& context) -> Ellipse
{
    auto blendRadius = [](const auto& radiusA, const auto& radiusB, const BlendingContext& context) -> Ellipse::RadialSize {
        return std::visit(WTF::makeVisitor(
            [&](const Ellipse::Length& lengthA, const Ellipse::Length& lengthB) -> Ellipse::RadialSize {
                return WebCore::Style::blend(lengthA, lengthB, context);
            },
            [&](const auto& a, const auto&) -> Ellipse::RadialSize {
                return a;
            }
        ), radiusA, radiusB);
    };

    return {
        .radii = SpaceSeparatedPair<Ellipse::RadialSize> {
            blendRadius(get<0>(a.radii), get<0>(b.radii), context),
            blendRadius(get<1>(a.radii), get<1>(b.radii), context)
        },
        .position = WebCore::Style::blend(a.position, b.position, context),
    };
}

} // namespace Style
} // namespace WebCore
