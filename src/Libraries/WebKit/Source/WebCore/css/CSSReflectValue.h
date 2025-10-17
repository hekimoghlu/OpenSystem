/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

class CSSReflectValue final : public CSSValue {
public:
    static Ref<CSSReflectValue> create(CSSValueID direction, Ref<CSSPrimitiveValue> offset, RefPtr<CSSValue> mask);

    CSSValueID direction() const { return m_direction; }
    const CSSPrimitiveValue& offset() const { return m_offset.get(); }
    const CSSValue* mask() const { return m_mask.get(); }

    String customCSSText() const;
    bool equals(const CSSReflectValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (func(m_offset.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        if (m_mask) {
            if (func(*m_mask) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    }

private:
    CSSReflectValue(CSSValueID direction, Ref<CSSPrimitiveValue> offset, RefPtr<CSSValue> mask);

    CSSValueID m_direction;
    Ref<CSSPrimitiveValue> m_offset;
    RefPtr<CSSValue> m_mask;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSReflectValue, isReflectValue())
