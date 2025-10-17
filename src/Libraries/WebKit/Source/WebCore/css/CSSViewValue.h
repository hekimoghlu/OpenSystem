/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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

// Class containing the value of a view() function, as used in animation-timeline:
// https://drafts.csswg.org/scroll-animations-1/#funcdef-view.
class CSSViewValue final : public CSSValue {
public:
    static Ref<CSSViewValue> create()
    {
        return adoptRef(*new CSSViewValue(nullptr, nullptr, nullptr));
    }

    static Ref<CSSViewValue> create(RefPtr<CSSValue>&& axis, RefPtr<CSSValue>&& startInset, RefPtr<CSSValue>&& endInset)
    {
        return adoptRef(*new CSSViewValue(WTFMove(axis), WTFMove(startInset), WTFMove(endInset)));
    }

    String customCSSText() const;

    const RefPtr<CSSValue>& axis() const { return m_axis; }
    const RefPtr<CSSValue>& startInset() const { return m_startInset; }
    const RefPtr<CSSValue>& endInset() const { return m_endInset; }

    bool equals(const CSSViewValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (m_axis) {
            if (func(*m_axis) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        if (m_startInset) {
            if (func(*m_startInset) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        if (m_endInset) {
            if (func(*m_endInset) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    }


private:
    CSSViewValue(RefPtr<CSSValue>&& axis, RefPtr<CSSValue>&& startInset, RefPtr<CSSValue>&& endInset)
        : CSSValue(ClassType::View)
        , m_axis(WTFMove(axis))
        , m_startInset(WTFMove(startInset))
        , m_endInset(WTFMove(endInset))
    {
    }

    RefPtr<CSSValue> m_axis;
    RefPtr<CSSValue> m_startInset;
    RefPtr<CSSValue> m_endInset;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSViewValue, isViewValue())
