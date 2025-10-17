/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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
#include <wtf/Function.h>

namespace WebCore {

class StyleImage;

namespace Style {
class BuilderState;
}

class CSSCrossfadeValue final : public CSSValue {
public:
    static Ref<CSSCrossfadeValue> create(Ref<CSSValue>&& fromValueOrNone, Ref<CSSValue>&& toValueOrNone, Ref<CSSPrimitiveValue>&& percentageValue, bool isPrefixed = false);

    ~CSSCrossfadeValue();

    bool equals(const CSSCrossfadeValue&) const;
    bool equalInputImages(const CSSCrossfadeValue&) const;

    String customCSSText() const;
    bool isPrefixed() const { return m_isPrefixed; }

    RefPtr<StyleImage> createStyleImage(const Style::BuilderState&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (func(m_fromValueOrNone.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        if (func(m_toValueOrNone.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        if (func(m_percentageValue.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        return IterationStatus::Continue;
    }

private:
    CSSCrossfadeValue(Ref<CSSValue>&& fromValueOrNone, Ref<CSSValue>&& toValueOrNone, Ref<CSSPrimitiveValue>&& percentageValue, bool isPrefixed);

    Ref<CSSValue> m_fromValueOrNone;
    Ref<CSSValue> m_toValueOrNone;
    Ref<CSSPrimitiveValue> m_percentageValue;
    bool m_isPrefixed;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSCrossfadeValue, isCrossfadeValue())
