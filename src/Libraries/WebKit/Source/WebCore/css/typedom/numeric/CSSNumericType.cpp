/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#include "CSSNumericType.h"

#include "CSSNumericValue.h"
#include "CSSUnits.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

std::optional<CSSNumericType> CSSNumericType::create(CSSUnitType unit, int exponent)
{
    // https://drafts.css-houdini.org/css-typed-om/#cssnumericvalue-create-a-type
    CSSNumericType type;
    switch (unitCategory(unit)) {
    case CSSUnitCategory::Number:
        return { WTFMove(type) };
    case CSSUnitCategory::Percent:
        type.percent = exponent;
        return { WTFMove(type) };
    case CSSUnitCategory::AbsoluteLength:
    case CSSUnitCategory::FontRelativeLength:
    case CSSUnitCategory::ViewportPercentageLength:
        type.length = exponent;
        return { WTFMove(type) };
    case CSSUnitCategory::Angle:
        type.angle = exponent;
        return { WTFMove(type) };
    case CSSUnitCategory::Time:
        type.time = exponent;
        return { WTFMove(type) };
    case CSSUnitCategory::Frequency:
        type.frequency = exponent;
        return { WTFMove(type) };
    case CSSUnitCategory::Resolution:
        type.resolution = exponent;
        return { WTFMove(type) };
    case CSSUnitCategory::Flex:
        type.flex = exponent;
        return { WTFMove(type) };
    case CSSUnitCategory::Other:
        break;
    }
    
    return std::nullopt;
}

std::optional<CSSNumericType> CSSNumericType::addTypes(CSSNumericType a, CSSNumericType b)
{
    // https://drafts.css-houdini.org/css-typed-om/#cssnumericvalue-add-two-types
    if (a.percentHint && b.percentHint && *a.percentHint != *b.percentHint)
        return std::nullopt;

    if (a.percentHint)
        b.applyPercentHint(*a.percentHint);
    if (b.percentHint)
        a.applyPercentHint(*b.percentHint);

    if (a == b)
        return { WTFMove(a) };

    for (auto type : eachBaseType()) {
        if (type == CSSNumericBaseType::Percent)
            continue;
        if (!a.valueForType(type) && !b.valueForType(type))
            continue;
        a.applyPercentHint(type);
        b.applyPercentHint(type);
        if (a.valueForType(type) != b.valueForType(type))
            return std::nullopt;
    }

    return { WTFMove(a) };
}

template<typename Argument> std::optional<CSSNumericType> typeFromVector(const Vector<Ref<CSSNumericValue>>& values, std::optional<CSSNumericType>(*function)(Argument, Argument))
{
    if (values.isEmpty())
        return std::nullopt;
    std::optional<CSSNumericType> type = values[0]->type();
    for (size_t i = 1; i < values.size(); ++i) {
        type = function(*type, values[i]->type());
        if (!type)
            return std::nullopt;
    }
    return type;
}

std::optional<CSSNumericType> CSSNumericType::addTypes(const Vector<Ref<CSSNumericValue>>& values)
{
    return typeFromVector(values, addTypes);
}

std::optional<CSSNumericType> CSSNumericType::multiplyTypes(const CSSNumericType& a, const CSSNumericType& b)
{
    // https://drafts.css-houdini.org/css-typed-om/#cssnumericvalue-multiply-two-types
    if (a.percentHint && b.percentHint && *a.percentHint != *b.percentHint)
        return std::nullopt;

    auto add = [] (auto left, auto right) -> CSSNumericType::BaseTypeStorage {
        if (!left)
            return right;
        if (!right)
            return left;
        return *left + *right;
    };
    
    return { {
        add(a.length, b.length),
        add(a.angle, b.angle),
        add(a.time, b.time),
        add(a.frequency, b.frequency),
        add(a.resolution, b.resolution),
        add(a.flex, b.flex),
        add(a.percent, b.percent),
        a.percentHint ? a.percentHint : b.percentHint
    } };
}

std::optional<CSSNumericType> CSSNumericType::multiplyTypes(const Vector<Ref<CSSNumericValue>>& values)
{
    return typeFromVector(values, multiplyTypes);
}

String CSSNumericType::debugString() const
{
    return makeString('{',
        length ? makeString(" length:"_s, *length) : String(),
        angle ? makeString(" angle:"_s, *angle) : String(),
        time ? makeString(" time:"_s, *time) : String(),
        frequency ? makeString(" frequency:"_s, *frequency) : String(),
        resolution ? makeString(" resolution:"_s, *resolution) : String(),
        flex ? makeString(" flex:"_s, *flex) : String(),
        percent ? makeString(" percent:"_s, *percent) : String(),
        percentHint ? makeString(" percentHint:"_s, WebCore::debugString(*percentHint)) : String(),
    " }"_s);
}

auto CSSNumericType::valueForType(CSSNumericBaseType type) -> BaseTypeStorage&
{
    switch (type) {
    case CSSNumericBaseType::Length:
        return length;
    case CSSNumericBaseType::Angle:
        return angle;
    case CSSNumericBaseType::Time:
        return time;
    case CSSNumericBaseType::Frequency:
        return frequency;
    case CSSNumericBaseType::Resolution:
        return resolution;
    case CSSNumericBaseType::Flex:
        return flex;
    case CSSNumericBaseType::Percent:
        return percent;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void CSSNumericType::applyPercentHint(CSSNumericBaseType hint)
{
    // https://drafts.css-houdini.org/css-typed-om/#apply-the-percent-hint
    auto& optional = valueForType(hint);
    if (!optional)
        optional = 0;
    if (percent)
        *optional += *std::exchange(percent, 0);
    percentHint = hint;
}

size_t CSSNumericType::nonZeroEntryCount() const
{
    size_t count { 0 };
    count += length && *length;
    count += angle && *angle;
    count += time && *time;
    count += frequency && *frequency;
    count += resolution && *resolution;
    count += flex && *flex;
    count += percent && *percent;
    return count;
}

} // namespace WebCore
