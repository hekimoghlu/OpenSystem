/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#include "StyleInsetFunction.h"

#include "FloatRect.h"
#include "GeometryUtilities.h"
#include "Path.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"
#include <wtf/TinyLRUCache.h>

namespace WebCore {
namespace Style {

// MARK: - Path Caching

struct RoundedInsetPathPolicy : public TinyLRUCachePolicy<FloatRoundedRect, WebCore::Path> {
public:
    static bool isKeyNull(const FloatRoundedRect& rect)
    {
        return rect.isEmpty();
    }

    static WebCore::Path createValueForKey(const FloatRoundedRect& rect)
    {
        WebCore::Path path;
        path.addRoundedRect(rect, PathRoundedRect::Strategy::PreferBezier);
        return path;
    }
};

static const WebCore::Path& cachedRoundedInsetPath(const FloatRoundedRect& rect)
{
    static NeverDestroyed<TinyLRUCache<FloatRoundedRect, WebCore::Path, 4, RoundedInsetPathPolicy>> cache;
    return cache.get().get(rect);
}

// MARK: - Path

WebCore::Path PathComputation<Inset>::operator()(const Inset& value, const FloatRect& boundingBox)
{
    auto boundingSize = boundingBox.size();

    auto left = evaluate(value.insets.left(), boundingSize.width());
    auto top = evaluate(value.insets.top(), boundingSize.height());
    auto rect = FloatRect {
        left + boundingBox.x(),
        top + boundingBox.y(),
        std::max<float>(boundingSize.width() - left - evaluate(value.insets.right(), boundingSize.width()), 0),
        std::max<float>(boundingSize.height() - top - evaluate(value.insets.bottom(), boundingSize.height()), 0)
    };

    auto radii = evaluate(value.radii, boundingSize);
    radii.scale(calcBorderRadiiConstraintScaleFor(rect, radii));

    return cachedRoundedInsetPath(FloatRoundedRect { rect, radii });
}

} // namespace Style
} // namespace WebCore
