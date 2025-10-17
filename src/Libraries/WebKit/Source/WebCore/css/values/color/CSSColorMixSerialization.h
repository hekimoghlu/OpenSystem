/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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

#include "CSSColorMix.h"
#include "StyleColorMix.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

bool isCalc(const ColorMix::Component::Percentage&);
constexpr bool isCalc(const Style::ColorMix::Component::Percentage&) { return false; }

bool is50Percent(const ColorMix::Component::Percentage&);
bool is50Percent(const Style::ColorMix::Component::Percentage&);

bool sumTo100Percent(const ColorMix::Component::Percentage&, const ColorMix::Component::Percentage&);
bool sumTo100Percent(const Style::ColorMix::Component::Percentage&, const Style::ColorMix::Component::Percentage&);

std::optional<PercentageRaw<>> subtractFrom100Percent(const ColorMix::Component::Percentage&);
std::optional<PercentageRaw<>> subtractFrom100Percent(const Style::ColorMix::Component::Percentage&);

void serializeColorMixColor(StringBuilder&, const ColorMix::Component&);
void serializeColorMixColor(StringBuilder&, const Style::ColorMix::Component&);

void serializeColorMixPercentage(StringBuilder&, const ColorMix::Component::Percentage&);
void serializeColorMixPercentage(StringBuilder&, const Style::ColorMix::Component::Percentage&);

template<typename ColorMixType>
void serializationForColorMixPercentage1(StringBuilder& builder, const ColorMixType& colorMix)
{
    if (colorMix.mixComponents1.percentage && colorMix.mixComponents2.percentage) {
        if (is50Percent(*colorMix.mixComponents1.percentage) && is50Percent(*colorMix.mixComponents2.percentage))
            return;
        builder.append(' ');
        serializeColorMixPercentage(builder, *colorMix.mixComponents1.percentage);
    } else if (colorMix.mixComponents1.percentage) {
        if (is50Percent(*colorMix.mixComponents1.percentage))
            return;
        builder.append(' ');
        serializeColorMixPercentage(builder, *colorMix.mixComponents1.percentage);
    } else if (colorMix.mixComponents2.percentage) {
        if (is50Percent(*colorMix.mixComponents2.percentage))
            return;

        auto subtractedPercent = subtractFrom100Percent(*colorMix.mixComponents2.percentage);
        if (!subtractedPercent)
            return;

        builder.append(' ');
        serializationForCSS(builder, *subtractedPercent);
    }
}

template<typename ColorMixType>
void serializationForColorMixPercentage2(StringBuilder& builder, const ColorMixType& colorMix)
{
    if (colorMix.mixComponents1.percentage && colorMix.mixComponents2.percentage) {
        if (sumTo100Percent(*colorMix.mixComponents1.percentage, *colorMix.mixComponents2.percentage))
            return;

        builder.append(' ');
        serializeColorMixPercentage(builder, *colorMix.mixComponents2.percentage);
    } else if (colorMix.mixComponents2.percentage) {
        if (is50Percent(*colorMix.mixComponents2.percentage))
            return;
        if (!isCalc(*colorMix.mixComponents2.percentage))
            return;

        builder.append(' ');
        serializeColorMixPercentage(builder, *colorMix.mixComponents2.percentage);
    }
}

// https://drafts.csswg.org/css-color-5/#serial-color-mix
template<typename ColorMixType>
void serializationForCSSColorMix(StringBuilder& builder, const ColorMixType& colorMix)
{
    builder.append("color-mix(in "_s);
    serializationForCSS(builder, colorMix.colorInterpolationMethod);
    builder.append(", "_s);
    serializeColorMixColor(builder, colorMix.mixComponents1);
    serializationForColorMixPercentage1(builder, colorMix);
    builder.append(", "_s);
    serializeColorMixColor(builder, colorMix.mixComponents2);
    serializationForColorMixPercentage2(builder, colorMix);
    builder.append(')');
}

} // namespace CSS
} // namespace WebCore
