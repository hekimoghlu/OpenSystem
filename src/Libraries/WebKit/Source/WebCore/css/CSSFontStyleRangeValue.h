/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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

#include "CSSPrimitiveValue.h"
#include "CSSValue.h"
#include "CSSValueList.h"

namespace WebCore {

class CSSFontStyleRangeValue final : public CSSValue {
public:
    static Ref<CSSFontStyleRangeValue> create(Ref<CSSPrimitiveValue>&& fontStyleValue)
    {
        return adoptRef(*new CSSFontStyleRangeValue(WTFMove(fontStyleValue), nullptr));
    }
    static Ref<CSSFontStyleRangeValue> create(Ref<CSSPrimitiveValue>&& fontStyleValue, RefPtr<CSSValueList>&& obliqueValues)
    {
        return adoptRef(*new CSSFontStyleRangeValue(WTFMove(fontStyleValue), WTFMove(obliqueValues)));
    }

    String customCSSText() const;

    bool equals(const CSSFontStyleRangeValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (func(fontStyleValue.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        if (obliqueValues) {
            if (func(*obliqueValues) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    }

    Ref<CSSPrimitiveValue> fontStyleValue;
    RefPtr<CSSValueList> obliqueValues;

private:
    CSSFontStyleRangeValue(Ref<CSSPrimitiveValue>&& fontStyleValue, RefPtr<CSSValueList>&& obliqueValues)
        : CSSValue(ClassType::FontStyleRange)
        , fontStyleValue(WTFMove(fontStyleValue))
        , obliqueValues(WTFMove(obliqueValues))
    {
    }
};

}

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSFontStyleRangeValue, isFontStyleRangeValue())
