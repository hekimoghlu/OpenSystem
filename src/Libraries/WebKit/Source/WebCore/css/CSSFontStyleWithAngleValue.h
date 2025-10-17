/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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

namespace WebCore {

// This class is currently only used for oblique. If we use it for more styles in the future we'll need to store the keyword.
class CSSFontStyleWithAngleValue final : public CSSValue {
public:
    static Ref<CSSFontStyleWithAngleValue> create(Ref<CSSPrimitiveValue>&& obliqueAngle);

    const CSSPrimitiveValue& obliqueAngle() const { return m_obliqueAngle; }
    Ref<CSSPrimitiveValue> protectedObliqueAngle() const { return m_obliqueAngle; }

    String customCSSText() const;
    bool equals(const CSSFontStyleWithAngleValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (func(m_obliqueAngle.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        return IterationStatus::Continue;
    }

private:
    CSSFontStyleWithAngleValue(Ref<CSSPrimitiveValue>&& obliqueAngle);

    Ref<CSSPrimitiveValue> m_obliqueAngle;
};

}

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSFontStyleWithAngleValue, isFontStyleWithAngleValue())
