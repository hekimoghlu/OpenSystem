/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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

#include "CSSColorDescriptors.h"
#include "CSSRelativeColor.h"
#include "CSSRelativeColorResolver.h"
#include "CSSRelativeColorSerialization.h"
#include "Color.h"
#include "ColorSerialization.h"
#include "StyleColor.h"
#include "StyleResolvedColor.h"
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace Style {

template<typename D, unsigned Index> using RelativeColorComponent = GetCSSColorParseTypeWithCalcAndSymbolsComponentResult<D, Index>;

template<typename D> struct RelativeColor {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    using Descriptor = D;

    Style::Color origin;
    CSSColorParseTypeWithCalcAndSymbols<Descriptor> components;

    bool operator==(const RelativeColor<Descriptor>&) const = default;
};

template<typename D> bool operator==(const UniqueRef<RelativeColor<D>>& a, const UniqueRef<RelativeColor<D>>& b)
{
    return a.get() == b.get();
}

template<typename D> Style::Color toStyleColor(const CSS::RelativeColor<D>& unresolved, ColorResolutionState& state)
{
    ColorResolutionStateNester nester { state };

    auto origin = toStyleColor(unresolved.origin, state);
    if (!origin.isResolvedColor()) {
        // If the origin is not absolute, we cannot fully resolve the color yet.
        // Instead, we simplify the calc values using the conversion data, and
        // return a Style::RelativeColor to be resolved at use time.
        return Style::Color {
            RelativeColor<D> {
                .origin = WTFMove(origin),
                .components = simplifyUnevaluatedCalc(unresolved.components, state.conversionData, CSSCalcSymbolTable { })
            }
        };
    }

    // If the origin is absolute, we can fully resolve the entire color.
    auto color = resolve(
        CSS::RelativeColorResolver<D> {
            .origin = origin.resolvedColor(),
            .components = unresolved.components
        },
        state.conversionData
    );

    return { ResolvedColor { WTFMove(color) } };
}

template<typename D> WebCore::Color resolveColor(const RelativeColor<D>& relative, const WebCore::Color& currentColor)
{
    return resolveNoConversionDataRequired(
        CSS::RelativeColorResolver<D> {
            .origin = relative.origin.resolveColor(currentColor),
            .components = relative.components
        }
    );
}

template<typename D> bool containsCurrentColor(const RelativeColor<D>& relative)
{
    return WebCore::Style::containsCurrentColor(relative.origin);
}

template<typename D> void serializationForCSS(StringBuilder& builder, const RelativeColor<D>& relative)
{
    CSS::serializationForCSSRelativeColor(builder, relative);
}

template<typename D> String serializationForCSS(const RelativeColor<D>& relative)
{
    StringBuilder builder;
    serializationForCSS(builder, relative);
    return builder.toString();
}

template<typename D> WTF::TextStream& operator<<(WTF::TextStream& ts, const RelativeColor<D>& relative)
{
    ts << "relativeColor(" << serializationForCSS(relative) << ")";
    return ts;
}

} // namespace Style
} // namespace WebCore
