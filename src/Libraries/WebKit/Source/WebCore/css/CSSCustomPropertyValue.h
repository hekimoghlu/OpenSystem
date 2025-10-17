/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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

#include "CSSValue.h"
#include "CSSVariableData.h"
#include "CSSVariableReferenceValue.h"
#include "Length.h"
#include "StyleColor.h"
#include "StyleImage.h"
#include "TransformOperation.h"
#include <wtf/PointerComparison.h>
#include <wtf/URL.h>

namespace WebCore {

class CSSParserToken;

class CSSCustomPropertyValue final : public CSSValue {
public:
    struct NumericSyntaxValue {
        double value;
        CSSUnitType unitType;

        bool operator==(const NumericSyntaxValue&) const = default;
    };

    struct TransformSyntaxValue {
        Ref<TransformOperation> transform;
        bool operator==(const TransformSyntaxValue& other) const { return transform.get() == other.transform.get(); }
    };

    using SyntaxValue = std::variant<Length, NumericSyntaxValue, Style::Color, RefPtr<StyleImage>, URL, String, TransformSyntaxValue>;

    struct SyntaxValueList {
        Vector<SyntaxValue> values;
        ValueSeparator separator;

        friend bool operator==(const SyntaxValueList&, const SyntaxValueList&) = default;
    };

    using VariantValue = std::variant<Ref<CSSVariableReferenceValue>, CSSValueID, Ref<CSSVariableData>, SyntaxValue, SyntaxValueList>;

    static Ref<CSSCustomPropertyValue> createEmpty(const AtomString& name);

    static Ref<CSSCustomPropertyValue> createUnresolved(const AtomString& name, Ref<CSSVariableReferenceValue>&& value)
    {
        return adoptRef(*new CSSCustomPropertyValue(name, VariantValue { std::in_place_type<Ref<CSSVariableReferenceValue>>, WTFMove(value) }));
    }

    static Ref<CSSCustomPropertyValue> createWithID(const AtomString& name, CSSValueID);

    static Ref<CSSCustomPropertyValue> createSyntaxAll(const AtomString& name, Ref<CSSVariableData>&& value)
    {
        return adoptRef(*new CSSCustomPropertyValue(name, VariantValue { std::in_place_type<Ref<CSSVariableData>>, WTFMove(value) }));
    }

    static Ref<CSSCustomPropertyValue> createForSyntaxValue(const AtomString& name, SyntaxValue&& syntaxValue)
    {
        return adoptRef(*new CSSCustomPropertyValue(name, VariantValue { WTFMove(syntaxValue) }));
    }

    static Ref<CSSCustomPropertyValue> createForSyntaxValueList(const AtomString& name, SyntaxValueList&& syntaxValueList)
    {
        return adoptRef(*new CSSCustomPropertyValue(name, VariantValue { WTFMove(syntaxValueList) }));
    }

    String customCSSText() const;

    const AtomString& name() const { return m_name; }
    bool isResolved() const { return !std::holds_alternative<Ref<CSSVariableReferenceValue>>(m_value); }
    bool isUnset() const { return std::holds_alternative<CSSValueID>(m_value) && std::get<CSSValueID>(m_value) == CSSValueUnset; }
    bool isInvalid() const { return std::holds_alternative<CSSValueID>(m_value) && std::get<CSSValueID>(m_value) == CSSValueInvalid; }
    bool isInherit() const { return std::holds_alternative<CSSValueID>(m_value) && std::get<CSSValueID>(m_value) == CSSValueInherit; }
    bool isCurrentColor() const;
    bool containsCSSWideKeyword() const;
    bool isAnimatable() const;

    const VariantValue& value() const { return m_value; }

    const Vector<CSSParserToken>& tokens() const;

    bool equals(const CSSCustomPropertyValue&) const;

    Ref<const CSSVariableData> asVariableData() const;

    bool customMayDependOnBaseURL() const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (auto* value = std::get_if<Ref<CSSVariableReferenceValue>>(&m_value)) {
            if (func(*value) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    }

private:
    CSSCustomPropertyValue(const AtomString& name, VariantValue&& value)
        : CSSValue(ClassType::CustomProperty)
        , m_name(name)
        , m_value(WTFMove(value))
    {
    }

    const AtomString m_name;
    const VariantValue m_value;
    mutable String m_cachedCSSText;
    mutable RefPtr<CSSVariableData> m_cachedTokens;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSCustomPropertyValue, isCustomPropertyValue())
