/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include "MatchResult.h"
#include <array>
#include <wtf/TZoneMalloc.h>

namespace JSC {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(StringReplaceCache, WTF_INTERNAL);

class JSImmutableButterfly;
class RegExp;

class StringReplaceCache {
    WTF_MAKE_TZONE_ALLOCATED(StringReplaceCache);
    WTF_MAKE_NONCOPYABLE(StringReplaceCache);

public:
    static constexpr unsigned cacheSize = 64;

    StringReplaceCache() = default;

    struct Entry {
        RefPtr<AtomStringImpl> m_subject { nullptr };
        RegExp* m_regExp { nullptr };
        JSImmutableButterfly* m_result { nullptr }; // We use JSImmutableButterfly since we would like to keep all entries alive while repeatedly calling a JS function.
        MatchResult m_matchResult { };
        Vector<int> m_lastMatch { };
    };

    Entry* get(const String& subject, RegExp*);
    void set(const String& subject, RegExp*, JSImmutableButterfly*, MatchResult, const Vector<int>&);

    DECLARE_VISIT_AGGREGATE;

    void clear()
    {
        m_entries.fill(Entry { });
    }

private:
    std::array<Entry, cacheSize> m_entries { };
};

} // namespace JSC
