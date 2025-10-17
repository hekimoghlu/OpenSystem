/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#include "StylePathFunction.h"

#include "AffineTransform.h"
#include "FloatRect.h"
#include "GeometryUtilities.h"
#include "Path.h"
#include "RenderStyle.h"
#include "RenderStyleInlines.h"
#include "SVGPathUtilities.h"
#include "StylePrimitiveNumericTypes+Blending.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"
#include <wtf/TinyLRUCache.h>

namespace WebCore {
namespace Style {

// MARK: - Path Caching

struct SVGPathTransformedByteStream {
    bool isEmpty() const
    {
        return rawStream.isEmpty();
    }

    WebCore::Path path() const
    {
        auto path = buildPathFromByteStream(rawStream);
        if (zoom != 1)
            path.transform(AffineTransform().scale(zoom));
        path.translate(toFloatSize(offset));
        return path;
    }

    bool operator==(const SVGPathTransformedByteStream&) const = default;

    SVGPathByteStream rawStream;
    float zoom;
    FloatPoint offset;
};

struct TransformedByteStreamPathPolicy : TinyLRUCachePolicy<SVGPathTransformedByteStream, WebCore::Path> {
    static bool isKeyNull(const SVGPathTransformedByteStream& stream)
    {
        return stream.isEmpty();
    }

    static WebCore::Path createValueForKey(const SVGPathTransformedByteStream& stream)
    {
        return stream.path();
    }
};

static const WebCore::Path& cachedTransformedByteStreamPath(const SVGPathByteStream& stream, float zoom, const FloatPoint& offset)
{
    static NeverDestroyed<TinyLRUCache<SVGPathTransformedByteStream, WebCore::Path, 4, TransformedByteStreamPathPolicy>> cache;
    return cache.get().get(SVGPathTransformedByteStream { stream, zoom, offset });
}

// MARK: - Conversion

static SVGPathByteStream copySVGPathByteStream(const SVGPathByteStream& source, PathConversion conversion)
{
    if (conversion == PathConversion::ForceAbsolute) {
        // Only returns the resulting absolute path if the conversion succeeds.
        if (auto result = convertSVGPathByteStreamToAbsoluteCoordinates(source))
            return *result;
    }
    return source;
}

auto ToCSS<Path::Data>::operator()(const Path::Data& value, const RenderStyle&) -> CSS::Path::Data
{
    return { copySVGPathByteStream(value.byteStream, PathConversion::None) };
}

auto ToStyle<CSS::Path::Data>::operator()(const CSS::Path::Data& value, const BuilderState&) -> Path::Data
{
    return { copySVGPathByteStream(value.byteStream, PathConversion::None) };
}

auto ToCSS<Path>::operator()(const Path& value, const RenderStyle& style) -> CSS::Path
{
    return {
        .fillRule = toCSS(value.fillRule, style),
        .data = toCSS(value.data, style)
    };
}

auto ToStyle<CSS::Path>::operator()(const CSS::Path& value, const BuilderState& state) -> Path
{
    return {
        .fillRule = toStyle(value.fillRule, state),
        .data = toStyle(value.data, state),
        .zoom = 1
    };
}

auto overrideToCSS(const Style::PathFunction& path, const RenderStyle& style, PathConversion conversion) -> CSS::PathFunction
{
    return {
        .parameters = CSS::Path {
            .fillRule = Style::toCSS(path->fillRule, style),
            .data = { copySVGPathByteStream(path->data.byteStream, conversion) }
        }
    };
}

auto overrideToStyle(const CSS::PathFunction& path, const BuilderState& state, std::optional<float> zoom) -> PathFunction
{
    return {
        .parameters = Path {
            .fillRule = toStyle(path->fillRule, state),
            .data = toStyle(path->data, state),
            .zoom = zoom.value_or(state.style().usedZoom())
        }
    };
}

// MARK: - Path

WebCore::Path PathComputation<Path>::operator()(const Path& value, const FloatRect& boundingBox)
{
    return cachedTransformedByteStreamPath(value.data.byteStream, value.zoom, boundingBox.location());
}

// MARK: - Wind Rule

WebCore::WindRule WindRuleComputation<Path>::operator()(const Path& value)
{
    return (!value.fillRule || std::holds_alternative<CSS::Keyword::Nonzero>(*value.fillRule)) ? WindRule::NonZero : WindRule::EvenOdd;
}

// MARK: - Blending

auto Blending<Path>::canBlend(const Path& a, const Path& b) -> bool
{
    return windRule(a) == windRule(b)
        && canBlendSVGPathByteStreams(a.data.byteStream, b.data.byteStream);
}

auto Blending<Path>::blend(const Path& a, const Path& b, const BlendingContext& context) -> Path
{
    SVGPathByteStream resultingPathBytes;
    buildAnimatedSVGPathByteStream(a.data.byteStream, b.data.byteStream, resultingPathBytes, context.progress);

    return {
        .fillRule = a.fillRule,
        .data = { WTFMove(resultingPathBytes) },
        .zoom = a.zoom
    };
}

} // namespace Style
} // namespace WebCore
