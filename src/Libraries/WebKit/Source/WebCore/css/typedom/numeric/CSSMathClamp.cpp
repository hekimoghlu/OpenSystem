/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#include "CSSMathClamp.h"

#include "CSSCalcTree.h"
#include "CSSNumericValue.h"
#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSMathClamp);

ExceptionOr<Ref<CSSMathClamp>> CSSMathClamp::create(CSSNumberish&& lower, CSSNumberish&& value, CSSNumberish&& upper)
{
    auto rectifiedLower = rectifyNumberish(WTFMove(lower));
    auto rectifiedValue = rectifyNumberish(WTFMove(value));
    auto rectifiedUpper = rectifyNumberish(WTFMove(upper));

    auto addedType = CSSNumericType::addTypes(rectifiedLower->type(), rectifiedValue->type());
    if (!addedType)
        return Exception { ExceptionCode::TypeError };
    addedType = CSSNumericType::addTypes(*addedType, rectifiedUpper->type());
    if (!addedType)
        return Exception { ExceptionCode::TypeError };

    return adoptRef(*new CSSMathClamp(WTFMove(*addedType), WTFMove(rectifiedLower), WTFMove(rectifiedValue), WTFMove(rectifiedUpper)));
}

CSSMathClamp::CSSMathClamp(CSSNumericType&& type, Ref<CSSNumericValue>&& lower, Ref<CSSNumericValue>&& value, Ref<CSSNumericValue>&& upper)
    : CSSMathValue(WTFMove(type))
    , m_lower(WTFMove(lower))
    , m_value(WTFMove(value))
    , m_upper(WTFMove(upper))
{
}

void CSSMathClamp::serialize(StringBuilder& builder, OptionSet<SerializationArguments>) const
{
    // https://drafts.css-houdini.org/css-typed-om/#calc-serialization
    OptionSet<SerializationArguments> serializationArguments { SerializationArguments::Nested, SerializationArguments::WithoutParentheses };
    builder.append("clamp("_s);
    m_lower->serialize(builder, serializationArguments);
    builder.append(", "_s);
    m_value->serialize(builder, serializationArguments);
    builder.append(", "_s);
    m_upper->serialize(builder, serializationArguments);
    builder.append(')');
}

auto CSSMathClamp::toSumValue() const -> std::optional<SumValue>
{
    auto validateSumValue = [](const std::optional<SumValue>& sumValue, const UnitMap* expectedUnits) {
        return sumValue && sumValue->size() == 1 && (!expectedUnits || *expectedUnits == sumValue->first().units);
    };

    auto lower = m_lower->toSumValue();
    if (!validateSumValue(lower, nullptr))
        return std::nullopt;
    auto value = m_value->toSumValue();
    if (!validateSumValue(value, &lower->first().units))
        return std::nullopt;
    auto upper = m_upper->toSumValue();
    if (!validateSumValue(upper, &lower->first().units))
        return std::nullopt;

    value->first().value = std::max(lower->first().value, std::min(value->first().value, upper->first().value));
    return value;
}

bool CSSMathClamp::equals(const CSSNumericValue& other) const
{
    // https://drafts.css-houdini.org/css-typed-om/#equal-numeric-value
    auto* otherClamp = dynamicDowncast<CSSMathClamp>(other);
    if (!otherClamp)
        return false;
    return m_lower->equals(otherClamp->m_lower)
        && m_value->equals(otherClamp->m_value)
        && m_upper->equals(otherClamp->m_upper);
}

std::optional<CSSCalc::Child> CSSMathClamp::toCalcTreeNode() const
{
    auto lower = m_lower->toCalcTreeNode();
    if (!lower)
        return std::nullopt;
    auto value = m_value->toCalcTreeNode();
    if (!value)
        return std::nullopt;
    auto upper = m_upper->toCalcTreeNode();
    if (!upper)
        return std::nullopt;

    auto clamp = CSSCalc::Clamp { .min = WTFMove(*lower), .val = WTFMove(*value), .max = WTFMove(*upper) };
    auto type = CSSCalc::toType(clamp);
    if (!type)
        return std::nullopt;

    return CSSCalc::makeChild(WTFMove(clamp), *type);
}

} // namespace WebCore
