/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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
#include "HTMLEntitySearch.h"

#include "HTMLEntityTable.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WebCore {

static const HTMLEntityTableEntry* midpoint(const HTMLEntityTableEntry* left, const HTMLEntityTableEntry* right)
{
    return &left[(right - left) / 2];
}

HTMLEntitySearch::HTMLEntitySearch()
    : m_first(HTMLEntityTable::firstEntry())
    , m_last(HTMLEntityTable::lastEntry())
{
}

HTMLEntitySearch::CompareResult HTMLEntitySearch::compare(const HTMLEntityTableEntry* entry, UChar nextCharacter) const
{
    UChar entryNextCharacter;
    if (entry->nameLengthExcludingSemicolon < m_currentLength + 1) {
        if (!entry->nameIncludesTrailingSemicolon || entry->nameLengthExcludingSemicolon < m_currentLength)
            return Before;
        entryNextCharacter = ';';
    } else
        entryNextCharacter = entry->nameCharacters()[m_currentLength];
    if (entryNextCharacter == nextCharacter)
        return Prefix;
    return entryNextCharacter < nextCharacter ? Before : After;
}

const HTMLEntityTableEntry* HTMLEntitySearch::findFirst(UChar nextCharacter) const
{
    auto* left = m_first;
    auto* right = m_last;
    if (left == right)
        return left;
    CompareResult result = compare(left, nextCharacter);
    if (result == Prefix)
        return left;
    if (result == After)
        return right;
    while (left + 1 < right) {
        auto* probe = midpoint(left, right);
        result = compare(probe, nextCharacter);
        if (result == Before)
            left = probe;
        else {
            ASSERT(result == After || result == Prefix);
            right = probe;
        }
    }
    ASSERT(left + 1 == right);
    return right;
}

const HTMLEntityTableEntry* HTMLEntitySearch::findLast(UChar nextCharacter) const
{
    auto* left = m_first;
    auto* right = m_last;
    if (left == right)
        return right;
    CompareResult result = compare(right, nextCharacter);
    if (result == Prefix)
        return right;
    if (result == Before)
        return left;
    while (left + 1 < right) {
        auto* probe = midpoint(left, right);
        result = compare(probe, nextCharacter);
        if (result == After)
            right = probe;
        else {
            ASSERT(result == Before || result == Prefix);
            left = probe;
        }
    }
    ASSERT(left + 1 == right);
    return left;
}

void HTMLEntitySearch::advance(UChar nextCharacter)
{
    ASSERT(isEntityPrefix());
    if (!m_currentLength) {
        m_first = HTMLEntityTable::firstEntryStartingWith(nextCharacter);
        m_last = HTMLEntityTable::lastEntryStartingWith(nextCharacter);
        if (!m_first || !m_last)
            return fail();
    } else {
        m_first = findFirst(nextCharacter);
        m_last = findLast(nextCharacter);
        if (m_first == m_last && compare(m_first, nextCharacter) != Prefix)
            return fail();
    }
    ++m_currentLength;
    if (m_first->nameLength() != m_currentLength)
        return;
    m_mostRecentMatch = m_first;
}

}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
