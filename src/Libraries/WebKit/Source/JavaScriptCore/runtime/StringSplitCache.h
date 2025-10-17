/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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

#include <array>
#include <wtf/DebugHeap.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(StringSplitCache, WTF_INTERNAL);

class JSImmutableButterfly;

class StringSplitCache {
    WTF_MAKE_TZONE_ALLOCATED(StringSplitCache);
    WTF_MAKE_NONCOPYABLE(StringSplitCache);
public:
    static constexpr unsigned cacheSize = 64;

    StringSplitCache() = default;

    struct Entry {
        RefPtr<AtomStringImpl> m_subject { nullptr };
        RefPtr<AtomStringImpl> m_separator { nullptr };
        JSImmutableButterfly* m_butterfly { nullptr };
    };

    JSImmutableButterfly* get(const String& subject, const String& separator);
    void set(const String& subject, const String& separator, JSImmutableButterfly*);

    void clear()
    {
        m_entries.fill(Entry { });
    }

private:
    std::array<Entry, cacheSize> m_entries { };
};

} // namespace JSC
