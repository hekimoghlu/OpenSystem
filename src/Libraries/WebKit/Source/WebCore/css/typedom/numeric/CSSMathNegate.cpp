/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#include "CSSMathNegate.h"

#include "CSSCalcTree.h"
#include "CSSNumericValue.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSMathNegate);

static CSSNumericType copyType(const CSSNumberish& numberish)
{
    return WTF::switchOn(numberish,
        [] (double) { return CSSNumericType(); },
        [] (const RefPtr<CSSNumericValue>& value) {
            if (!value)
                return CSSNumericType();
            return value->type();
        }
    );
}

CSSMathNegate::CSSMathNegate(CSSNumberish&& numberish)
    : CSSMathValue(copyType(numberish))
    , m_value(rectifyNumberish(WTFMove(numberish)))
{
}

void CSSMathNegate::serialize(StringBuilder& builder, OptionSet<SerializationArguments> arguments) const
{
    // https://drafts.css-houdini.org/css-typed-om/#calc-serialization
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(arguments.contains(SerializationArguments::Nested) ? "("_s : "calc("_s);
    builder.append('-');
    m_value->serialize(builder, arguments);
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(')');
}

auto CSSMathNegate::toSumValue() const -> std::optional<SumValue>
{
    // https://drafts.css-houdini.org/css-typed-om/#create-a-sum-value
    auto values = m_value->toSumValue();
    if (!values)
        return std::nullopt;
    for (auto& value : *values)
        value.value = value.value * -1;
    return values;
}

bool CSSMathNegate::equals(const CSSNumericValue& other) const
{
    // https://drafts.css-houdini.org/css-typed-om/#equal-numeric-value
    auto* otherNegate = dynamicDowncast<CSSMathNegate>(other);
    if (!otherNegate)
        return false;
    return m_value->equals(otherNegate->value());
}

std::optional<CSSCalc::Child> CSSMathNegate::toCalcTreeNode() const
{
    auto child = m_value->toCalcTreeNode();
    if (!child)
        return std::nullopt;

    auto negate = CSSCalc::Negate { .a = WTFMove(*child) };
    auto type = CSSCalc::toType(negate);
    if (!type)
        return std::nullopt;

    return CSSCalc::makeChild(WTFMove(negate), *type);
}

} // namespace WebCore
