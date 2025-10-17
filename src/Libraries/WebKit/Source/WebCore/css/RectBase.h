/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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

class RectBase {
public:
    const CSSPrimitiveValue& top() const { return m_top.get(); }
    const CSSPrimitiveValue& right() const { return m_right.get(); }
    const CSSPrimitiveValue& bottom() const { return m_bottom.get(); }
    const CSSPrimitiveValue& left() const { return m_left.get(); }

    bool equals(const RectBase& other) const
    {
        return compareCSSValue(m_top, other.m_top)
            && compareCSSValue(m_right, other.m_right)
            && compareCSSValue(m_left, other.m_left)
            && compareCSSValue(m_bottom, other.m_bottom);
    }

protected:
    explicit RectBase(Ref<CSSPrimitiveValue> value)
        : m_top(value)
        , m_right(value)
        , m_bottom(value)
        , m_left(WTFMove(value))
    { }
    RectBase(Ref<CSSPrimitiveValue> top, Ref<CSSPrimitiveValue> right, Ref<CSSPrimitiveValue> bottom, Ref<CSSPrimitiveValue> left)
        : m_top(WTFMove(top))
        , m_right(WTFMove(right))
        , m_bottom(WTFMove(bottom))
        , m_left(WTFMove(left))
    { }
    ~RectBase() = default;

private:
    Ref<const CSSPrimitiveValue> m_top;
    Ref<const CSSPrimitiveValue> m_right;
    Ref<const CSSPrimitiveValue> m_bottom;
    Ref<const CSSPrimitiveValue> m_left;
};

} // namespace WebCore
