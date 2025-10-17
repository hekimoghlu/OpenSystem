/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#include "CSSMathSum.h"

#include "CSSCalcTree.h"
#include "CSSMathNegate.h"
#include "CSSNumericArray.h"
#include "ExceptionOr.h"
#include <wtf/Algorithms.h>
#include <wtf/FixedVector.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSMathSum);

ExceptionOr<Ref<CSSMathSum>> CSSMathSum::create(FixedVector<CSSNumberish> numberishes)
{
    return create(WTF::map(WTFMove(numberishes), rectifyNumberish));
}

ExceptionOr<Ref<CSSMathSum>> CSSMathSum::create(Vector<Ref<CSSNumericValue>> values)
{
    if (values.isEmpty())
        return Exception { ExceptionCode::SyntaxError };

    auto type = CSSNumericType::addTypes(values);
    if (!type)
        return Exception { ExceptionCode::TypeError };

    return adoptRef(*new CSSMathSum(WTFMove(values), WTFMove(*type)));
}

CSSMathSum::CSSMathSum(Vector<Ref<CSSNumericValue>> values, CSSNumericType type)
    : CSSMathValue(WTFMove(type))
    , m_values(CSSNumericArray::create(WTFMove(values)))
{
}

void CSSMathSum::serialize(StringBuilder& builder, OptionSet<SerializationArguments> arguments) const
{
    // https://drafts.css-houdini.org/css-typed-om/#calc-serialization
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(arguments.contains(SerializationArguments::Nested) ? "("_s : "calc("_s);
    m_values->forEach([&](auto& numericValue, bool first) {
        OptionSet<SerializationArguments> operandSerializationArguments { SerializationArguments::Nested };
        operandSerializationArguments.set(SerializationArguments::WithoutParentheses, arguments.contains(SerializationArguments::WithoutParentheses));
        if (!first) {
            if (auto* mathNegate = dynamicDowncast<CSSMathNegate>(numericValue)) {
                builder.append(" - "_s);
                mathNegate->value().serialize(builder, operandSerializationArguments);
                return;
            }
            builder.append(" + "_s);
        }
        numericValue.serialize(builder, operandSerializationArguments);
    });
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(')');
}

auto CSSMathSum::toSumValue() const -> std::optional<SumValue>
{
    auto convertToNumericType = [] (const UnitMap& units) -> std::optional<CSSNumericType> {
        // https://drafts.css-houdini.org/css-typed-om/#create-a-type-from-a-unit-map
        CSSNumericType type;
        for (auto& pair : units) {
            auto unit = CSSNumericType::create(pair.key, pair.value);
            if (!unit)
                return std::nullopt;
            auto multipliedType = CSSNumericType::multiplyTypes(type, *unit);
            if (!multipliedType)
                return std::nullopt;
            type = WTFMove(*multipliedType);
        }
        return type;
    };

    // https://drafts.css-houdini.org/css-typed-om/#create-a-sum-value
    SumValue values;
    for (auto& item : m_values->array()) {
        auto value = item->toSumValue();
        if (!value)
            return std::nullopt;
        for (auto& subvalue : *value) {
            auto index = values.findIf([&](auto& value) {
                return value.units == subvalue.units;
            });
            if (index == notFound)
                values.append(WTFMove(subvalue));
            else
                values[index].value += subvalue.value;
        }
    }

    auto type = convertToNumericType(values[0].units);
    if (!type)
        return std::nullopt;
    for (size_t i = 1; i < values.size(); ++i) {
        auto thisType = convertToNumericType(values[i].units);
        if (!thisType)
            return std::nullopt;
        type = CSSNumericType::addTypes(*type, *thisType);
        if (!type)
            return std::nullopt;
    }
    
    return { WTFMove(values) };
}

std::optional<CSSCalc::Child> CSSMathSum::toCalcTreeNode() const
{
    CSSCalc::Children children = WTF::compactMap(m_values->array(), [](auto& child) {
        return child->toCalcTreeNode();
    });
    if (children.size() != m_values->array().size())
        return std::nullopt;

    auto sum = CSSCalc::Sum { .children = WTFMove(children) };
    auto type = CSSCalc::toType(sum);
    if (!type)
        return std::nullopt;

    return CSSCalc::makeChild(WTFMove(sum), *type);
}

} // namespace WebCore
