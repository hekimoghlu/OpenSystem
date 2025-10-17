/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#include "CSSMathInvert.h"

#include "CSSCalcTree.h"
#include "CSSNumericValue.h"
#include "CSSPrimitiveValue.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSMathInvert);

Ref<CSSMathInvert> CSSMathInvert::create(CSSNumberish&& numberish)
{
    return adoptRef(*new CSSMathInvert(WTFMove(numberish)));
}

static CSSNumericType negatedType(const CSSNumberish& numberish)
{
    // https://drafts.css-houdini.org/css-typed-om/#type-of-a-cssmathvalue
    return WTF::switchOn(numberish,
        [] (double) { return CSSNumericType(); },
        [] (const RefPtr<CSSNumericValue>& value) {
            if (!value)
                return CSSNumericType();
            CSSNumericType type = value->type();
            auto negate = [] (auto& optional) {
                if (optional)
                    optional = *optional * -1;
            };
            negate(type.length);
            negate(type.angle);
            negate(type.time);
            negate(type.frequency);
            negate(type.resolution);
            negate(type.flex);
            negate(type.percent);
            return type;
        }
    );
}

CSSMathInvert::CSSMathInvert(CSSNumberish&& numberish)
    : CSSMathValue(negatedType(numberish))
    , m_value(rectifyNumberish(WTFMove(numberish)))
{
}

void CSSMathInvert::serialize(StringBuilder& builder, OptionSet<SerializationArguments> arguments) const
{
    // https://drafts.css-houdini.org/css-typed-om/#calc-serialization
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(arguments.contains(SerializationArguments::Nested) ? "("_s : "calc("_s);
    builder.append("1 / "_s);
    m_value->serialize(builder, arguments);
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(')');
}

auto CSSMathInvert::toSumValue() const -> std::optional<SumValue>
{
    // https://drafts.css-houdini.org/css-typed-om/#create-a-sum-value
    auto values = m_value->toSumValue();
    if (!values)
        return std::nullopt;
    if (values->size() != 1)
        return std::nullopt;
    auto& value = (*values)[0];
    if (!value.value)
        return std::nullopt;
    value.value = 1.0 / value.value;

    UnitMap negatedExponents;
    for (auto& pair : value.units)
        negatedExponents.add(pair.key, -1 * pair.value);
    value.units = WTFMove(negatedExponents);

    return values;
}

bool CSSMathInvert::equals(const CSSNumericValue& other) const
{
    // https://drafts.css-houdini.org/css-typed-om/#equal-numeric-value
    auto* otherInvert = dynamicDowncast<CSSMathInvert>(other);
    if (!otherInvert)
        return false;
    return m_value->equals(otherInvert->value());
}

std::optional<CSSCalc::Child> CSSMathInvert::toCalcTreeNode() const
{
    auto child = m_value->toCalcTreeNode();
    if (!child)
        return std::nullopt;

    auto invert = CSSCalc::Invert { .a = WTFMove(*child) };
    auto type = CSSCalc::toType(invert);
    if (!type)
        return std::nullopt;

    return CSSCalc::makeChild(WTFMove(invert), *type);
}

} // namespace WebCore
