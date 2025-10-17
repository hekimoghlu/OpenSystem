/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
#include "CSSNumericArray.h"
#include "CSSNumericValue.h"
#include "CSSStyleValue.h"

namespace WebCore {

class CSSMathValue : public CSSNumericValue {
public:
    CSSMathValue(CSSNumericType type)
        : CSSNumericValue(WTFMove(type))
    {
    }

    virtual CSSMathOperator getOperator() const = 0;

    template<typename T> bool equalsImpl(const CSSNumericValue&) const;

    RefPtr<CSSValue> toCSSValue() const final;
};

template<typename T> bool CSSMathValue::equalsImpl(const CSSNumericValue& other) const
{
    // https://drafts.css-houdini.org/css-typed-om/#equal-numeric-value
    auto* otherT = dynamicDowncast<T>(other);
    if (!otherT)
        return false;

    ASSERT(getType() == other.getType());
    auto& thisValues = static_cast<const T*>(this)->values();
    auto& otherValues = otherT->values();
    auto length = thisValues.length();
    if (length != otherValues.length())
        return false;

    for (size_t i = 0 ; i < length; ++i) {
        if (!thisValues.array()[i]->equals(otherValues.array()[i].get()))
            return false;
    }

    return true;
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSMathValue)
static bool isType(const WebCore::CSSStyleValue& styleValue) { return WebCore::isCSSMathValue(styleValue.getType()); }
SPECIALIZE_TYPE_TRAITS_END()
