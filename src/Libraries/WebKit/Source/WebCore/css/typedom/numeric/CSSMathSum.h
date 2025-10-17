/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "CSSMathOperator.h"
#include "CSSMathValue.h"
#include "CSSNumericValue.h"

namespace WebCore {

class CSSNumericArray;

class CSSMathSum final : public CSSMathValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSMathSum);
public:
    static ExceptionOr<Ref<CSSMathSum>> create(FixedVector<CSSNumberish>);
    static ExceptionOr<Ref<CSSMathSum>> create(Vector<Ref<CSSNumericValue>>);
    const CSSNumericArray& values() const { return m_values.get(); }

    std::optional<CSSCalc::Child> toCalcTreeNode() const final;

private:
    CSSMathOperator getOperator() const final { return CSSMathOperator::Sum; }
    CSSStyleValueType getType() const override { return CSSStyleValueType::CSSMathSum; }
    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const final;
    std::optional<SumValue> toSumValue() const final;
    bool equals(const CSSNumericValue& other) const final { return equalsImpl<CSSMathSum>(other); }

    CSSMathSum(Vector<Ref<CSSNumericValue>>, CSSNumericType);
    Ref<CSSNumericArray> m_values;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSMathSum)
static bool isType(const WebCore::CSSStyleValue& styleValue) { return styleValue.getType() == WebCore::CSSStyleValueType::CSSMathSum; }
static bool isType(const WebCore::CSSNumericValue& numericValue) { return numericValue.getType() == WebCore::CSSStyleValueType::CSSMathSum; }
static bool isType(const WebCore::CSSMathValue& mathValue) { return mathValue.getType() == WebCore::CSSStyleValueType::CSSMathSum; }
SPECIALIZE_TYPE_TRAITS_END()
