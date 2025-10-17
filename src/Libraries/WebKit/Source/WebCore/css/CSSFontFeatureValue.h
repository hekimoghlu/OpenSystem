/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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
#include "FontTaggedSettings.h"

namespace WebCore {

class CSSFontFeatureValue final : public CSSValue {
public:
    static Ref<CSSFontFeatureValue> create(FontTag&& tag, Ref<CSSPrimitiveValue>&& value)
    {
        return adoptRef(*new CSSFontFeatureValue(WTFMove(tag), WTFMove(value)));
    }

    const FontTag& tag() const { return m_tag; }
    const CSSPrimitiveValue& value() const { return m_value; }
    Ref<CSSPrimitiveValue> protectedValue() const { return m_value; }
    String customCSSText() const;

    bool equals(const CSSFontFeatureValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (func(m_value.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        return IterationStatus::Continue;
    }

private:
    CSSFontFeatureValue(FontTag&&, Ref<CSSPrimitiveValue>&&);

    FontTag m_tag;
    Ref<CSSPrimitiveValue> m_value;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSFontFeatureValue, isFontFeatureValue())
