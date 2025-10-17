/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

class CSSBorderImageSliceValue final : public CSSValue {
public:
    static Ref<CSSBorderImageSliceValue> create(Quad, bool fill);
    ~CSSBorderImageSliceValue();

    const Quad& slices() const { return m_slices; }
    bool fill() const { return m_fill; }

    String customCSSText() const;
    bool equals(const CSSBorderImageSliceValue&) const;

private:
    CSSBorderImageSliceValue(Quad, bool fill);

    // These four values are used to make "cuts" in the border image. They can be numbers
    // or percentages.
    Quad m_slices;
    bool m_fill { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSBorderImageSliceValue, isBorderImageSliceValue())
