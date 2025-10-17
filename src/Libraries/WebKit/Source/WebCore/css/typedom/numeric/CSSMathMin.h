/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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
#include "CSSNumericValue.h"

namespace WebCore {

class CSSNumericArray;

class CSSMathMin final : public CSSMathValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSMathMin);
public:
    static ExceptionOr<Ref<CSSMathMin>> create(FixedVector<CSSNumberish>&&);
    static ExceptionOr<Ref<CSSMathMin>> create(Vector<Ref<CSSNumericValue>>&&);
    const CSSNumericArray& values() const;

    std::optional<CSSCalc::Child> toCalcTreeNode() const final;

private:
    CSSMathOperator getOperator() const final { return CSSMathOperator::Min; }
    CSSStyleValueType getType() const final { return CSSStyleValueType::CSSMathMin; }
    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const final;
    std::optional<SumValue> toSumValue() const final;
    bool equals(const CSSNumericValue& other) const final { return equalsImpl<CSSMathMin>(other); }

    CSSMathMin(Vector<Ref<CSSNumericValue>>&&, CSSNumericType&&);
    Ref<CSSNumericArray> m_values;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSMathMin)
static bool isType(const WebCore::CSSStyleValue& styleValue) { return styleValue.getType() == WebCore::CSSStyleValueType::CSSMathMin; }
static bool isType(const WebCore::CSSNumericValue& numericValue) { return numericValue.getType() == WebCore::CSSStyleValueType::CSSMathMin; }
static bool isType(const WebCore::CSSMathValue& mathValue) { return mathValue.getType() == WebCore::CSSStyleValueType::CSSMathMin; }
SPECIALIZE_TYPE_TRAITS_END()
