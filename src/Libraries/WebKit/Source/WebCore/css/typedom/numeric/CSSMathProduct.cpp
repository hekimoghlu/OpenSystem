/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include "CSSMathProduct.h"

#include "CSSCalcTree.h"
#include "CSSMathInvert.h"
#include "CSSNumericArray.h"
#include "ExceptionOr.h"
#include <wtf/FixedVector.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSMathProduct);

ExceptionOr<Ref<CSSMathProduct>> CSSMathProduct::create(FixedVector<CSSNumberish> numberishes)
{
    return create(WTF::map(WTFMove(numberishes), rectifyNumberish));
}

ExceptionOr<Ref<CSSMathProduct>> CSSMathProduct::create(Vector<Ref<CSSNumericValue>> values)
{
    if (values.isEmpty())
        return Exception { ExceptionCode::SyntaxError };

    auto type = CSSNumericType::multiplyTypes(values);
    if (!type)
        return Exception { ExceptionCode::TypeError };

    return adoptRef(*new CSSMathProduct(WTFMove(values), WTFMove(*type)));
}

CSSMathProduct::CSSMathProduct(Vector<Ref<CSSNumericValue>> values, CSSNumericType type)
    : CSSMathValue(WTFMove(type))
    , m_values(CSSNumericArray::create(WTFMove(values)))
{
}

void CSSMathProduct::serialize(StringBuilder& builder, OptionSet<SerializationArguments> arguments) const
{
    // https://drafts.css-houdini.org/css-typed-om/#calc-serialization
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(arguments.contains(SerializationArguments::Nested) ? "("_s : "calc("_s);
    m_values->forEach([&](auto& numericValue, bool first) {
        OptionSet<SerializationArguments> operandSerializationArguments { SerializationArguments::Nested };
        operandSerializationArguments.set(SerializationArguments::WithoutParentheses, arguments.contains(SerializationArguments::WithoutParentheses));
        if (!first) {
            if (auto* mathNegate = dynamicDowncast<CSSMathInvert>(numericValue)) {
                builder.append(" / "_s);
                mathNegate->value().serialize(builder, operandSerializationArguments);
                return;
            }
            builder.append(" * "_s);
        }
        numericValue.serialize(builder, operandSerializationArguments);
    });
    if (!arguments.contains(SerializationArguments::WithoutParentheses))
        builder.append(')');
}

auto CSSMathProduct::toSumValue() const -> std::optional<SumValue>
{
    auto productOfUnits = [] (const auto& units1, const auto& units2) {
        // https://drafts.css-houdini.org/css-typed-om/#product-of-two-unit-maps
        auto result = units1;
        for (auto& pair : units2) {
            auto addResult = result.add(pair.key, pair.value);
            if (!addResult.isNewEntry)
                addResult.iterator->value += pair.value;
            if (!addResult.iterator->value)
                result.remove(pair.key);
        }
        return result;
    };
    
    // https://drafts.css-houdini.org/css-typed-om/#create-a-sum-value
    SumValue values { Addend { 1.0, { } } };
    for (auto& item : m_values->array()) {
        auto newValues = item->toSumValue();
        if (!newValues)
            return std::nullopt;
        SumValue temp;
        for (auto& item1 : values) {
            for (auto& item2 : *newValues) {
                Addend item { item1.value * item2.value, productOfUnits(item1.units, item2.units) };
                temp.append(WTFMove(item));
            }
        }
        values = WTFMove(temp);
    }
    return { WTFMove(values) };
}

std::optional<CSSCalc::Child> CSSMathProduct::toCalcTreeNode() const
{
    CSSCalc::Children children = WTF::compactMap(m_values->array(), [](auto& child) {
        return child->toCalcTreeNode();
    });
    if (children.size() != m_values->array().size())
        return std::nullopt;

    auto product = CSSCalc::Product { .children = WTFMove(children) };
    auto type = CSSCalc::toType(product);
    if (!type)
        return std::nullopt;

    return CSSCalc::makeChild(WTFMove(product), *type);
}

} // namespace WebCore
