/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "ColorSerialization.h"
#include <optional>
#include <variant>
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

// https://drafts.csswg.org/css-color-5/#serial-relative-color
template<typename RelativeColorType>
void serializationForCSSRelativeColor(StringBuilder& builder, const RelativeColorType& relativeColor)
{
    using Descriptor = typename RelativeColorType::Descriptor;
    using ColorType = typename Descriptor::ColorType;

    if constexpr (Descriptor::usesColorFunctionForSerialization) {
        builder.append("color(from "_s);
        serializationForCSS(builder, relativeColor.origin);
        builder.append(' ');
        builder.append(serialization(ColorSpaceFor<ColorType>));
    } else {
        builder.append(Descriptor::serializationFunctionName);
        builder.append("(from "_s);
        serializationForCSS(builder, relativeColor.origin);
    }

    auto [c1, c2, c3, alpha] = relativeColor.components;

    builder.append(' ');
    serializationForCSS(builder, c1);
    builder.append(' ');
    serializationForCSS(builder, c2);
    builder.append(' ');
    serializationForCSS(builder, c3);

    if (alpha) {
        builder.append(" / "_s);
        serializationForCSS(builder, *alpha);
    }

    builder.append(')');
}

} // namespace CSS
} // namespace WebCore
