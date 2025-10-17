/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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

namespace WebCore {

class CSSOffsetRotateValue final : public CSSValue {
public:
    static Ref<CSSOffsetRotateValue> create(RefPtr<CSSPrimitiveValue>&& modifier, RefPtr<CSSPrimitiveValue>&& angle)
    {
        return adoptRef(*new CSSOffsetRotateValue(WTFMove(modifier), WTFMove(angle)));
    }

    String customCSSText() const;

    CSSPrimitiveValue* modifier() const { return m_modifier.get(); }
    CSSPrimitiveValue* angle() const { return m_angle.get(); }

    bool isInitialValue() const;

    bool equals(const CSSOffsetRotateValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (m_modifier) {
            if (func(*m_modifier) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        if (m_angle) {
            if (func(*m_angle) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    }

private:
    CSSOffsetRotateValue(RefPtr<CSSPrimitiveValue>&& modifier, RefPtr<CSSPrimitiveValue>&& angle)
        : CSSValue(ClassType::OffsetRotate)
        , m_modifier(WTFMove(modifier))
        , m_angle(WTFMove(angle))
    {
        ASSERT(m_modifier || m_angle);

        if (m_modifier)
            ASSERT(m_modifier->isValueID());

        if (m_angle)
            ASSERT(m_angle->isAngle());
    }

    RefPtr<CSSPrimitiveValue> m_modifier;
    RefPtr<CSSPrimitiveValue> m_angle;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSOffsetRotateValue, isOffsetRotateValue())
