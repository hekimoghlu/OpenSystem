/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#include "StringReplaceCache.h"

namespace JSC {

inline StringReplaceCache::Entry* StringReplaceCache::get(const String& subject, RegExp* regExp)
{
    DisallowGC disallowGC;
    if (!subject.impl() || !subject.impl()->isAtom())
        return nullptr;
    ASSERT(regExp->global());
    ASSERT(subject.length() >= Options::thresholdForStringReplaceCache());

    auto* subjectImpl = static_cast<AtomStringImpl*>(subject.impl());
    unsigned index = subjectImpl->hash() & (cacheSize - 1);
    {
        auto& entry = m_entries[index];
        if (entry.m_subject == subjectImpl && entry.m_regExp == regExp)
            return &entry;
    }
    {
        auto& entry = m_entries[(index + 1) & (cacheSize - 1)];
        if (entry.m_subject == subjectImpl && entry.m_regExp == regExp)
            return &entry;
    }
    return nullptr;
}

inline void StringReplaceCache::set(const String& subject, RegExp* regExp, JSImmutableButterfly* result, MatchResult matchResult, const Vector<int>& lastMatch)
{
    DisallowGC disallowGC;
    if (!subject.impl() || !subject.impl()->isAtom())
        return;

    auto* subjectImpl = static_cast<AtomStringImpl*>(subject.impl());
    unsigned index = subjectImpl->hash() & (cacheSize - 1);
    {
        auto& entry1 = m_entries[index];
        if (!entry1.m_subject) {
            entry1.m_subject = subjectImpl;
            entry1.m_regExp = regExp;
            entry1.m_lastMatch = lastMatch;
            entry1.m_matchResult = matchResult;
            entry1.m_result = result;
        } else {
            auto& entry2 = m_entries[(index + 1) & (cacheSize - 1)];
            if (!entry2.m_subject) {
                entry2.m_subject = subjectImpl;
                entry2.m_regExp = regExp;
                entry2.m_lastMatch = lastMatch;
                entry2.m_matchResult = matchResult;
                entry2.m_result = result;
            } else {
                entry2 = { };
                entry1.m_subject = subjectImpl;
                entry1.m_regExp = regExp;
                entry1.m_lastMatch = lastMatch;
                entry1.m_matchResult = matchResult;
                entry1.m_result = result;
            }
        }
    }
}

template<typename Visitor>
inline void StringReplaceCache::visitAggregateImpl(Visitor& visitor)
{
    for (auto& entry : m_entries) {
        visitor.appendUnbarriered(entry.m_regExp);
        visitor.appendUnbarriered(entry.m_result);
    }
}

DEFINE_VISIT_AGGREGATE(StringReplaceCache);

} // namespace JSC
