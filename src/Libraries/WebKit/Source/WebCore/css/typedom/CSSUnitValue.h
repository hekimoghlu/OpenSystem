/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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

#include "CSSNumericValue.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class CSSUnitType : uint8_t;

class CSSUnitValue final : public CSSNumericValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSUnitValue);
public:
    static ExceptionOr<Ref<CSSUnitValue>> create(double value, const String& unit);
    static Ref<CSSUnitValue> create(double value, CSSUnitType unit) { return adoptRef(*new CSSUnitValue(value, unit)); }

    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const final;

    double value() const { return m_value; }
    void setValue(double value) { m_value = value; }
    ASCIILiteral unit() const;
    ASCIILiteral unitSerialization() const;
    CSSUnitType unitEnum() const { return m_unit; }

    RefPtr<CSSUnitValue> convertTo(CSSUnitType) const;
    static CSSUnitType parseUnit(const String& unit);

    RefPtr<CSSValue> toCSSValue() const final;
    RefPtr<CSSValue> toCSSValueWithProperty(CSSPropertyID) const final;
    std::optional<CSSCalc::Child> toCalcTreeNode() const final;

private:
    CSSUnitValue(double, CSSUnitType);

    CSSStyleValueType getType() const final { return CSSStyleValueType::CSSUnitValue; }
    std::optional<SumValue> toSumValue() const final;
    bool equals(const CSSNumericValue&) const final;

    double m_value;
    const CSSUnitType m_unit;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSUnitValue)
static bool isType(const WebCore::CSSStyleValue& styleValue) { return styleValue.getType() == WebCore::CSSStyleValueType::CSSUnitValue; }
SPECIALIZE_TYPE_TRAITS_END()
