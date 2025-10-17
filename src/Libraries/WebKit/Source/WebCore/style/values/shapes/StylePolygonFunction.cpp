/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "StylePolygonFunction.h"

#include "FloatRect.h"
#include "GeometryUtilities.h"
#include "Path.h"
#include "StylePrimitiveNumericTypes+Blending.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"
#include <wtf/TinyLRUCache.h>

namespace WebCore {
namespace Style {

// MARK: - Path Caching

struct PolygonPathPolicy : TinyLRUCachePolicy<Vector<FloatPoint>, WebCore::Path> {
public:
    static bool isKeyNull(const Vector<FloatPoint>& points)
    {
        return !points.size();
    }

    static WebCore::Path createValueForKey(const Vector<FloatPoint>& points)
    {
        return WebCore::Path(points);
    }
};

static const WebCore::Path& cachedPolygonPath(const Vector<FloatPoint>& points)
{
    static NeverDestroyed<TinyLRUCache<Vector<FloatPoint>, WebCore::Path, 4, PolygonPathPolicy>> cache;
    return cache.get().get(points);
}

// MARK: - Path

WebCore::Path PathComputation<Polygon>::operator()(const Polygon& value, const FloatRect& boundingBox)
{
    auto boundingLocation = boundingBox.location();
    auto boundingSize = boundingBox.size();
    auto points = value.vertices.value.map([&](const auto& vertex) {
        return evaluate(vertex, boundingSize) + boundingLocation;
    });
    return cachedPolygonPath(points);
}

// MARK: - Wind Rule

WebCore::WindRule WindRuleComputation<Polygon>::operator()(const Polygon& value)
{
    return (!value.fillRule || std::holds_alternative<CSS::Keyword::Nonzero>(*value.fillRule)) ? WindRule::NonZero : WindRule::EvenOdd;
}

// MARK: - Blending

auto Blending<Polygon>::canBlend(const Polygon& a, const Polygon& b) -> bool
{
    return windRule(a) == windRule(b)
        && a.vertices.size() == b.vertices.size();
}

auto Blending<Polygon>::blend(const Polygon& a, const Polygon& b, const BlendingContext& context) -> Polygon
{
    return {
        .fillRule = a.fillRule,
        .vertices = WebCore::Style::blend(a.vertices, b.vertices, context),
    };
}

} // namespace Style
} // namespace WebCore
