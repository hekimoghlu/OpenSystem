/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#include "RenderStyleConstants.h"

namespace WebCore {

// Class containing the value of a scroll() function, as used in animation-timeline:
// https://drafts.csswg.org/scroll-animations-1/#funcdef-scroll.
class CSSScrollValue final : public CSSValue {
public:
    static Ref<CSSScrollValue> create()
    {
        return adoptRef(*new CSSScrollValue(nullptr, nullptr));
    }

    static Ref<CSSScrollValue> create(RefPtr<CSSValue>&& scroller, RefPtr<CSSValue>&& axis)
    {
        return adoptRef(*new CSSScrollValue(WTFMove(scroller), WTFMove(axis)));
    }

    String customCSSText() const;

    const RefPtr<CSSValue>& scroller() const { return m_scroller; }
    const RefPtr<CSSValue>& axis() const { return m_axis; }

    bool equals(const CSSScrollValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (m_scroller) {
            if (func(*m_scroller) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        if (m_axis) {
            if (func(*m_axis) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    }

private:
    CSSScrollValue(RefPtr<CSSValue>&& scroller, RefPtr<CSSValue>&& axis)
        : CSSValue(ClassType::Scroll)
        , m_scroller(WTFMove(scroller))
        , m_axis(WTFMove(axis))
    {
    }

    RefPtr<CSSValue> m_scroller;
    RefPtr<CSSValue> m_axis;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSScrollValue, isScrollValue())
