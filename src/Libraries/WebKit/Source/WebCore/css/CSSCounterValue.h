/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#include <wtf/text/AtomString.h>

namespace WebCore {

class CSSCounterValue final : public CSSValue {
public:
    static Ref<CSSCounterValue> create(AtomString identifier, AtomString separator, RefPtr<CSSValue> counterStyle);

    const AtomString& identifier() const { return m_identifier; }
    const AtomString& separator() const { return m_separator; }
    RefPtr<CSSValue> counterStyle() const { return m_counterStyle; }
    String counterStyleCSSText() const;

    String customCSSText() const;
    bool equals(const CSSCounterValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (m_counterStyle) {
            if (func(*m_counterStyle) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    }

private:
    CSSCounterValue(AtomString identifier, AtomString separator, RefPtr<CSSValue> counterStyle);

    AtomString m_identifier;
    AtomString m_separator;
    RefPtr<CSSValue> m_counterStyle;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSCounterValue, isCounter())
