/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#include "Quad.h"

namespace WebCore {

class CSSPrimitiveValue;

class CSSBorderImageWidthValue final : public CSSValue {
public:
    static Ref<CSSBorderImageWidthValue> create(Quad, bool overridesBorderWidths);
    ~CSSBorderImageWidthValue();

    const Quad& widths() const { return m_widths; }
    bool overridesBorderWidths() const { return m_overridesBorderWidths; }

    String customCSSText() const;
    bool equals(const CSSBorderImageWidthValue&) const;

private:
    CSSBorderImageWidthValue(Quad, bool overridesBorderWidths);

    Quad m_widths;
    bool m_overridesBorderWidths { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSBorderImageWidthValue, isBorderImageWidthValue())
