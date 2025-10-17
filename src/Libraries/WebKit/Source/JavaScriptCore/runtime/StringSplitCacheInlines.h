/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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

#include "StringSplitCache.h"

namespace JSC {

inline JSImmutableButterfly* StringSplitCache::get(const String& subject, const String& separator)
{
    DisallowGC disallowGC;
    if (!subject.impl() || !subject.impl()->isAtom())
        return nullptr;
    if (!separator.impl() || !separator.impl()->isAtom())
        return nullptr;

    auto* subjectImpl = static_cast<AtomStringImpl*>(subject.impl());
    auto* separatorImpl = static_cast<AtomStringImpl*>(separator.impl());
    unsigned index = subjectImpl->hash() & (cacheSize - 1);
    {
        auto& entry = m_entries[index];
        if (entry.m_subject == subjectImpl && entry.m_separator == separatorImpl)
            return entry.m_butterfly;
    }
    {
        auto& entry = m_entries[(index + 1) & (cacheSize - 1)];
        if (entry.m_subject == subjectImpl && entry.m_separator == separatorImpl)
            return entry.m_butterfly;
    }
    return nullptr;
}

inline void StringSplitCache::set(const String& subject, const String& separator, JSImmutableButterfly* butterfly)
{
    DisallowGC disallowGC;
    if (!subject.impl() || !subject.impl()->isAtom())
        return;
    if (!separator.impl() || !separator.impl()->isAtom())
        return;

    auto* subjectImpl = static_cast<AtomStringImpl*>(subject.impl());
    auto* separatorImpl = static_cast<AtomStringImpl*>(separator.impl());
    unsigned index = subjectImpl->hash() & (cacheSize - 1);
    {
        auto& entry1 = m_entries[index];
        if (!entry1.m_subject) {
            entry1.m_subject = subjectImpl;
            entry1.m_separator = separatorImpl;
            entry1.m_butterfly = butterfly;
        } else {
            auto& entry2 = m_entries[(index + 1) & (cacheSize - 1)];
            if (!entry2.m_subject) {
                entry2.m_subject = subjectImpl;
                entry2.m_separator = separatorImpl;
                entry2.m_butterfly = butterfly;
            } else {
                entry2.m_subject = nullptr;
                entry2.m_separator = nullptr;
                entry1.m_subject = subjectImpl;
                entry1.m_separator = separatorImpl;
                entry1.m_butterfly = butterfly;
            }
        }
    }
}

} // namespace JSC
