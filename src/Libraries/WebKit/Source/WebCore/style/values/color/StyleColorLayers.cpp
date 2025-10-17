/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#include "StyleColorLayers.h"

#include "CSSColorLayersResolver.h"
#include "CSSColorLayersSerialization.h"
#include "ColorSerialization.h"
#include "StyleBuilderState.h"
#include "StyleColorResolutionState.h"
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace Style {

// MARK: - Conversion

Color toStyleColor(const CSS::ColorLayers& unresolved, ColorResolutionState& state)
{
    ColorResolutionStateNester nester { state };

    auto colors = CommaSeparatedVector<Color> { unresolved.colors.map([&](auto& color) -> Color {
        return toStyleColor(color, state);
    }) };

    if (std::ranges::any_of(colors, [](auto& color) { return color.isResolvedColor(); })) {
        // If the any of the layer's colors are not resolved, we cannot fully resolve the
        // color yet. Instead we return a Style::ColorLayers to be resolved at use time.
        return Color {
            ColorLayers {
                .blendMode = unresolved.blendMode,
                .colors = WTFMove(colors)
            }
        };
    }

    auto resolver = CSS::ColorLayersResolver {
        .blendMode = unresolved.blendMode,
        // FIXME: This should be made into a lazy transformed range to avoid the unnecessary temporary allocation.
        .colors = colors.map([&](const auto& color) {
            return color.resolvedColor();
        })
    };

    return blendSourceOver(WTFMove(resolver));
}

// MARK: Resolve

WebCore::Color resolveColor(const ColorLayers& colorLayers, const WebCore::Color& currentColor)
{
    return blendSourceOver(
        CSS::ColorLayersResolver {
            .blendMode = colorLayers.blendMode,
            .colors = colorLayers.colors.map([&](auto& color) {
                return color.resolveColor(currentColor);
            })
        }
    );
}

// MARK: - Current Color

bool containsCurrentColor(const ColorLayers& colorLayers)
{
    return std::ranges::any_of(colorLayers.colors, [&](auto& color) {
        return WebCore::Style::containsCurrentColor(color);
    });
}

// MARK: - Serialization

void serializationForCSS(StringBuilder& builder, const ColorLayers& colorLayers)
{
    CSS::serializationForCSSColorLayers(builder, colorLayers);
}

String serializationForCSS(const ColorLayers& colorLayers)
{
    StringBuilder builder;
    serializationForCSS(builder, colorLayers);
    return builder.toString();
}

// MARK: - TextStream

WTF::TextStream& operator<<(WTF::TextStream& ts, const ColorLayers& colorLayers)
{
    return ts << serializationForCSS(colorLayers);
}

} // namespace Style
} // namespace WebCore
