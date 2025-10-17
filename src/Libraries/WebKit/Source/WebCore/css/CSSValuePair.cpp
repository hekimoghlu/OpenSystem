/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
#include "config.h"
#include "CSSValuePair.h"

#include <wtf/Hasher.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

CSSValuePair::CSSValuePair(ValueSeparator separator, Ref<CSSValue> first, Ref<CSSValue> second, IdenticalValueSerialization serialization)
    : CSSValue(ClassType::ValuePair)
    , m_coalesceIdenticalValues(serialization != IdenticalValueSerialization::DoNotCoalesce)
    , m_first(WTFMove(first))
    , m_second(WTFMove(second))
{
    m_valueSeparator = separator;
}

Ref<CSSValuePair> CSSValuePair::create(Ref<CSSValue> first, Ref<CSSValue> second)
{
    return adoptRef(*new CSSValuePair(SpaceSeparator, WTFMove(first), WTFMove(second), IdenticalValueSerialization::Coalesce));
}

Ref<CSSValuePair> CSSValuePair::createSlashSeparated(Ref<CSSValue> first, Ref<CSSValue> second)
{
    return adoptRef(*new CSSValuePair(SlashSeparator, WTFMove(first), WTFMove(second), IdenticalValueSerialization::DoNotCoalesce));
}

Ref<CSSValuePair> CSSValuePair::createNoncoalescing(Ref<CSSValue> first, Ref<CSSValue> second)
{
    return adoptRef(*new CSSValuePair(SpaceSeparator, WTFMove(first), WTFMove(second), IdenticalValueSerialization::DoNotCoalesce));
}

bool CSSValuePair::canBeCoalesced() const
{
    return m_coalesceIdenticalValues && m_first->equals(m_second);
}

String CSSValuePair::customCSSText() const
{
    String first = m_first->cssText();
    String second = m_second->cssText();
    if (m_coalesceIdenticalValues && first == second)
        return first;
    return makeString(first, separatorCSSText(), second);
}

bool CSSValuePair::equals(const CSSValuePair& other) const
{
    return m_valueSeparator == other.m_valueSeparator
        && m_coalesceIdenticalValues == other.m_coalesceIdenticalValues
        && m_first->equals(other.m_first)
        && m_second->equals(other.m_second);
}

bool CSSValuePair::addDerivedHash(Hasher& hasher) const
{
    add(hasher, m_valueSeparator, m_coalesceIdenticalValues);
    return m_first->addHash(hasher) && m_second->addHash(hasher);
}

} // namespace WebCore
