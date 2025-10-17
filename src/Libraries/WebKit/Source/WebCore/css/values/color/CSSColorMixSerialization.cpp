/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
#include "CSSColorMixSerialization.h"

#include "CSSPrimitiveNumericTypes+Serialization.h"

namespace WebCore {
namespace CSS {

bool isCalc(const ColorMix::Component::Percentage& percentage)
{
    return percentage.isCalc();
}

bool is50Percent(const ColorMix::Component::Percentage& percentage)
{
    return WTF::switchOn(percentage,
        [](const ColorMix::Component::Percentage::Raw& raw) { return raw.value == 50.0; },
        [](const ColorMix::Component::Percentage::Calc&) { return false; }
    );
}

bool is50Percent(const Style::ColorMix::Component::Percentage& percentage)
{
    return percentage.value == 50.0;
}

bool sumTo100Percent(const ColorMix::Component::Percentage& a, const ColorMix::Component::Percentage& b)
{
    if (a.isCalc() || b.isCalc())
        return false;

    return a.raw()->value + b.raw()->value == 100.0;
}

bool sumTo100Percent(const Style::ColorMix::Component::Percentage& a, const Style::ColorMix::Component::Percentage& b)
{
    return a.value + b.value == 100.0;
}

std::optional<PercentageRaw<>> subtractFrom100Percent(const ColorMix::Component::Percentage& percentage)
{
    using Percentage = ColorMix::Component::Percentage;

    return WTF::switchOn(percentage,
        [&](const Percentage::Raw& raw) -> std::optional<PercentageRaw<>> {
            return PercentageRaw<> { 100.0 - raw.value };
        },
        [&](const Percentage::Calc&) -> std::optional<PercentageRaw<>> {
            return std::nullopt;
        }
    );
}

std::optional<PercentageRaw<>> subtractFrom100Percent(const Style::ColorMix::Component::Percentage& percentage)
{
    return PercentageRaw<> { 100.0 - percentage.value };
}

void serializeColorMixColor(StringBuilder& builder, const ColorMix::Component& component)
{
    serializationForCSS(builder, component.color);
}

void serializeColorMixColor(StringBuilder& builder, const Style::ColorMix::Component& component)
{
    serializationForCSS(builder, component.color);
}

void serializeColorMixPercentage(StringBuilder& builder, const ColorMix::Component::Percentage& percentage)
{
    serializationForCSS(builder, percentage);
}

void serializeColorMixPercentage(StringBuilder& builder, const Style::ColorMix::Component::Percentage& percentage)
{
    serializationForCSS(builder, PercentageRaw<> { percentage.value });
}

} // namespace CSS
} // namespace WebCore
