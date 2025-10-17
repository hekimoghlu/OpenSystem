/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#include "CSSMathMax.h"

#include "CSSCalcTree.h"
#include "CSSNumericArray.h"
#include "ExceptionOr.h"
#include <wtf/Algorithms.h>
#include <wtf/FixedVector.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSMathMax);

ExceptionOr<Ref<CSSMathMax>> CSSMathMax::create(FixedVector<CSSNumberish>&& numberishes)
{
    return create(WTF::map(WTFMove(numberishes), rectifyNumberish));
}

ExceptionOr<Ref<CSSMathMax>> CSSMathMax::create(Vector<Ref<CSSNumericValue>>&& values)
{
    if (values.isEmpty())
        return Exception { ExceptionCode::SyntaxError };

    auto type = CSSNumericType::addTypes(values);
    if (!type)
        return Exception { ExceptionCode::TypeError };

    return adoptRef(*new CSSMathMax(WTFMove(values), WTFMove(*type)));
}

CSSMathMax::CSSMathMax(Vector<Ref<CSSNumericValue>>&& values, CSSNumericType&& type)
    : CSSMathValue(WTFMove(type))
    , m_values(CSSNumericArray::create(WTFMove(values)))
{
}

const CSSNumericArray& CSSMathMax::values() const
{
    return m_values.get();
}

void CSSMathMax::serialize(StringBuilder& builder, OptionSet<SerializationArguments> arguments) const
{
    // https://drafts.css-houdini.org/css-typed-om/#calc-serialization
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append("max("_s);
    m_values->forEach([&](auto& numericValue, bool first) {
        if (!first)
            builder.append(", "_s);
        numericValue.serialize(builder, { SerializationArguments::Nested, SerializationArguments::WithoutParentheses });
    });
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(')');
}

auto CSSMathMax::toSumValue() const -> std::optional<SumValue>
{
    // https://drafts.css-houdini.org/css-typed-om/#create-a-sum-value
    auto& valuesArray = m_values->array();
    std::optional<SumValue> currentMax = valuesArray[0]->toSumValue();
    if (!currentMax || currentMax->size() != 1)
        return std::nullopt;
    for (size_t i = 1; i < valuesArray.size(); ++i) {
        auto currentValue = valuesArray[i]->toSumValue();
        if (!currentValue
            || currentValue->size() != 1
            || (*currentValue)[0].units != (*currentMax)[0].units)
            return std::nullopt;
        if ((*currentValue)[0].value > (*currentMax)[0].value)
            currentMax = WTFMove(currentValue);
    }
    return currentMax;
}

std::optional<CSSCalc::Child> CSSMathMax::toCalcTreeNode() const
{
    CSSCalc::Children children = WTF::compactMap(m_values->array(), [](auto& child) {
        return child->toCalcTreeNode();
    });
    if (children.isEmpty())
        return std::nullopt;

    auto max = CSSCalc::Max { .children = WTFMove(children) };
    auto type = CSSCalc::toType(max);
    if (!type)
        return std::nullopt;

    return CSSCalc::makeChild(WTFMove(max), *type);
}

} // namespace WebCore
