/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include <wtf/Function.h>

namespace WebCore {

class CSSValuePair final : public CSSValue {
public:
    static Ref<CSSValuePair> create(Ref<CSSValue>, Ref<CSSValue>);
    static Ref<CSSValuePair> createSlashSeparated(Ref<CSSValue>, Ref<CSSValue>);
    static Ref<CSSValuePair> createNoncoalescing(Ref<CSSValue>, Ref<CSSValue>);

    const CSSValue& first() const { return m_first; }
    const CSSValue& second() const { return m_second; }
    Ref<CSSValue> protectedFirst() const { return m_first; }
    Ref<CSSValue> protectedSecond() const { return m_second; }

    String customCSSText() const;
    bool equals(const CSSValuePair&) const;
    bool canBeCoalesced() const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (func(m_first.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        if (func(m_second.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        return IterationStatus::Continue;
    }

private:
    friend bool CSSValue::addHash(Hasher&) const;

    enum class IdenticalValueSerialization : bool { DoNotCoalesce, Coalesce };
    CSSValuePair(ValueSeparator, Ref<CSSValue>, Ref<CSSValue>, IdenticalValueSerialization);

    bool addDerivedHash(Hasher&) const;

    // FIXME: Store coalesce bit in CSSValue to cut down on object size.
    bool m_coalesceIdenticalValues { true };
    Ref<CSSValue> m_first;
    Ref<CSSValue> m_second;
};

inline const CSSValue& CSSValue::first() const
{
    return downcast<CSSValuePair>(*this).first();
}

inline Ref<CSSValue> CSSValue::protectedFirst() const
{
    return downcast<CSSValuePair>(*this).protectedFirst();
}

inline const CSSValue& CSSValue::second() const
{
    return downcast<CSSValuePair>(*this).second();
}

inline Ref<CSSValue> CSSValue::protectedSecond() const
{
    return downcast<CSSValuePair>(*this).protectedSecond();
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSValuePair, isPair())
