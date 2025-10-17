/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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

#include "Rect.h"

namespace WebCore {

class CSSRectValue final : public CSSValue {
public:
    static Ref<CSSRectValue> create(Rect);

    const Rect& rect() const { return m_rect; }

    String customCSSText() const;
    bool equals(const CSSRectValue&) const;

private:
    explicit CSSRectValue(Rect);

    Rect m_rect;
};

inline const Rect& CSSValue::rect() const
{
    return downcast<CSSRectValue>(*this).rect();
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSRectValue, isRect())
