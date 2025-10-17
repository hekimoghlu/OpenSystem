/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#include "StyleColorMix.h"

#include "CSSColorMixResolver.h"
#include "CSSColorMixSerialization.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "ColorSerialization.h"
#include "StyleColorResolutionState.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace Style {

// MARK: - Conversion

Color toStyleColor(const CSS::ColorMix& unresolved, ColorResolutionState& state)
{
    ColorResolutionStateNester nester { state };

    auto component1Color = toStyleColor(unresolved.mixComponents1.color, state);
    auto component2Color = toStyleColor(unresolved.mixComponents2.color, state);

    auto percentage1 = toStyle(unresolved.mixComponents1.percentage, state.conversionData);
    auto percentage2 = toStyle(unresolved.mixComponents2.percentage, state.conversionData);

    if (!component1Color.isResolvedColor() || !component2Color.isResolvedColor()) {
        // If the either component is not resolved, we cannot fully resolve the color
        // yet. Instead, we resolve the calc values using the conversion data, and return
        // a Style::ColorMix to be resolved at use time.
        return Color {
            ColorMix {
                unresolved.colorInterpolationMethod,
                ColorMix::Component {
                    WTFMove(component1Color),
                    WTFMove(percentage1)
                },
                ColorMix::Component {
                    WTFMove(component2Color),
                    WTFMove(percentage2)
                }
            }
        };
    }

    return mix(
        CSS::ColorMixResolver {
            unresolved.colorInterpolationMethod,
            CSS::ColorMixResolver::Component {
                component1Color.resolvedColor(),
                WTFMove(percentage1)
            },
            CSS::ColorMixResolver::Component {
                component2Color.resolvedColor(),
                WTFMove(percentage2)
            }
        }
    );
}


// MARK: - Resolve

WebCore::Color resolveColor(const ColorMix& colorMix, const WebCore::Color& currentColor)
{
    return mix(
        CSS::ColorMixResolver {
            colorMix.colorInterpolationMethod,
            CSS::ColorMixResolver::Component {
                colorMix.mixComponents1.color.resolveColor(currentColor),
                colorMix.mixComponents1.percentage
            },
            CSS::ColorMixResolver::Component {
                colorMix.mixComponents2.color.resolveColor(currentColor),
                colorMix.mixComponents2.percentage
            }
        }
    );
}

// MARK: - Current Color

bool containsCurrentColor(const ColorMix& colorMix)
{
    return WebCore::Style::containsCurrentColor(colorMix.mixComponents1.color)
        || WebCore::Style::containsCurrentColor(colorMix.mixComponents2.color);
}

// MARK: - Serialization

void serializationForCSS(StringBuilder& builder, const ColorMix& colorMix)
{
    CSS::serializationForCSSColorMix(builder, colorMix);
}

String serializationForCSS(const ColorMix& colorMix)
{
    StringBuilder builder;
    serializationForCSS(builder, colorMix);
    return builder.toString();
}

// MARK: - TextStream

static WTF::TextStream& operator<<(WTF::TextStream& ts, const ColorMix::Component& component)
{
    ts << component.color;
    if (component.percentage)
        ts << " " << component.percentage->value << "%";
    return ts;
}

WTF::TextStream& operator<<(WTF::TextStream& ts, const ColorMix& colorMix)
{
    ts << "color-mix(";
    ts << "in " << colorMix.colorInterpolationMethod;
    ts << ", " << colorMix.mixComponents1;
    ts << ", " << colorMix.mixComponents2;
    ts << ")";

    return ts;
}

} // namespace Style
} // namespace WebCore
