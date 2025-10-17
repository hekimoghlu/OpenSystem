/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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

#include "CSSMathValue.h"

namespace WebCore {

class CSSMathClamp final : public CSSMathValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSMathClamp);
public:
    static ExceptionOr<Ref<CSSMathClamp>> create(CSSNumberish&&, CSSNumberish&&, CSSNumberish&&);
    const CSSNumericValue& lower() const { return m_lower.get(); }
    const CSSNumericValue& value() const { return m_value.get(); }
    const CSSNumericValue& upper() const { return m_upper.get(); }

    std::optional<CSSCalc::Child> toCalcTreeNode() const final;

private:
    CSSMathOperator getOperator() const final { return CSSMathOperator::Clamp; }
    CSSStyleValueType getType() const final { return CSSStyleValueType::CSSMathClamp; }
    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const final;
    std::optional<SumValue> toSumValue() const final;
    bool equals(const CSSNumericValue&) const final;

    CSSMathClamp(CSSNumericType&&, Ref<CSSNumericValue>&&, Ref<CSSNumericValue>&&, Ref<CSSNumericValue>&&);

    Ref<CSSNumericValue> m_lower;
    Ref<CSSNumericValue> m_value;
    Ref<CSSNumericValue> m_upper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSMathClamp)
static bool isType(const WebCore::CSSStyleValue& styleValue) { return styleValue.getType() == WebCore::CSSStyleValueType::CSSMathClamp; }
static bool isType(const WebCore::CSSNumericValue& numericValue) { return numericValue.getType() == WebCore::CSSStyleValueType::CSSMathClamp; }
static bool isType(const WebCore::CSSMathValue& mathValue) { return mathValue.getType() == WebCore::CSSStyleValueType::CSSMathClamp; }
SPECIALIZE_TYPE_TRAITS_END()
