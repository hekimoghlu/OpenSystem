/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#include "StyleContrastColor.h"

#include "CSSContrastColorResolver.h"
#include "CSSContrastColorSerialization.h"
#include "ColorSerialization.h"
#include "StyleBuilderState.h"
#include "StyleColorResolutionState.h"
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace Style {

// MARK: - Conversion

Color toStyleColor(const CSS::ContrastColor& unresolved, ColorResolutionState& state)
{
    ColorResolutionStateNester nester { state };

    auto color = toStyleColor(unresolved.color, state);
    if (!color.isResolvedColor()) {
        return Color {
            ContrastColor {
                WTFMove(color),
                unresolved.max
            }
        };
    }

    return resolve(
        CSS::ContrastColorResolver {
            color.resolvedColor(),
            unresolved.max
        }
    );
}


// MARK: - Resolve

WebCore::Color resolveColor(const ContrastColor& contrastColor, const WebCore::Color& currentColor)
{
    return resolve(
        CSS::ContrastColorResolver {
            contrastColor.color.resolveColor(currentColor),
            contrastColor.max
        }
    );
}

// MARK: - Current Color

bool containsCurrentColor(const ContrastColor& contrastColor)
{
    return WebCore::Style::containsCurrentColor(contrastColor.color);
}

// MARK: - Serialization

void serializationForCSS(StringBuilder& builder, const ContrastColor& contrastColor)
{
    CSS::serializationForCSSContrastColor(builder, contrastColor);
}

String serializationForCSS(const ContrastColor& contrastColor)
{
    StringBuilder builder;
    serializationForCSS(builder, contrastColor);
    return builder.toString();
}

// MARK: - TextStream

WTF::TextStream& operator<<(WTF::TextStream& ts, const ContrastColor& contrastColor)
{
    return ts << serializationForCSS(contrastColor);
}

} // namespace Style
} // namespace WebCore
