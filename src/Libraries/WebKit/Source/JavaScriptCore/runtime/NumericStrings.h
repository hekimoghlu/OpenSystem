/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#include <wtf/HashFunctions.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class JSString;
class SmallStrings;

class NumericStrings {
public:
    static const size_t cacheSize = 256;

    template<typename T>
    struct CacheEntry {
        T key;
        String value;
    };

    template<typename T>
    struct CacheEntryWithJSString {
        T key;
        String value;
        JSString* jsString { nullptr };
    };

    struct StringWithJSString {
        String value;
        JSString* jsString { nullptr };

        static constexpr ptrdiff_t offsetOfJSString() { return OBJECT_OFFSETOF(StringWithJSString, jsString); }
    };

    ALWAYS_INLINE const String& add(double d)
    {
        auto& entry = lookup(d);
        if (d == entry.key && !entry.value.isNull())
            return entry.value;
        entry.key = d;
        entry.value = String::number(d);
        entry.jsString = nullptr;
        return entry.value;
    }

    ALWAYS_INLINE const String& add(int i)
    {
        if (static_cast<unsigned>(i) < cacheSize)
            return lookupSmallString(static_cast<unsigned>(i)).value;
        auto& entry = lookup(i);
        if (i == entry.key && !entry.value.isNull())
            return entry.value;
        entry.key = i;
        entry.value = String::number(i);
        entry.jsString = nullptr;
        return entry.value;
    }

    ALWAYS_INLINE const String& add(unsigned i)
    {
        if (i < cacheSize)
            return lookupSmallString(static_cast<unsigned>(i)).value;
        auto& entry = lookup(i);
        if (i == entry.key && !entry.value.isNull())
            return entry.value;
        entry.key = i;
        entry.value = String::number(i);
        return entry.value;
    }

    JSString* addJSString(VM&, int);
    JSString* addJSString(VM&, double);

    void clearOnGarbageCollection()
    {
        for (auto& entry : m_intCache)
            entry.jsString = nullptr;
        for (auto& entry : m_doubleCache)
            entry.jsString = nullptr;
        // 0-9 are managed by SmallStrings. They never die.
        for (unsigned i = 10; i < m_smallIntCache.size(); ++i)
            m_smallIntCache[i].jsString = nullptr;
    }

    template<typename Visitor>
    void visitAggregate(Visitor& visitor)
    {
        for (auto& entry : m_intCache)
            visitor.appendUnbarriered(entry.jsString);
        for (auto& entry : m_doubleCache)
            visitor.appendUnbarriered(entry.jsString);
        // 0-9 are managed by SmallStrings. They never die.
        for (unsigned i = 10; i < m_smallIntCache.size(); ++i)
            visitor.appendUnbarriered(m_smallIntCache[i].jsString);
    }

    const StringWithJSString* smallIntCache() { return m_smallIntCache.data(); }

    void initializeSmallIntCache(VM&);

private:
    CacheEntryWithJSString<double>& lookup(double d) { return m_doubleCache[WTF::FloatHash<double>::hash(d) & (cacheSize - 1)]; }
    CacheEntryWithJSString<int>& lookup(int i) { return m_intCache[WTF::IntHash<int>::hash(i) & (cacheSize - 1)]; }
    CacheEntry<unsigned>& lookup(unsigned i) { return m_unsignedCache[WTF::IntHash<unsigned>::hash(i) & (cacheSize - 1)]; }
    ALWAYS_INLINE StringWithJSString& lookupSmallString(unsigned i)
    {
        ASSERT(i < cacheSize);
        if (m_smallIntCache[i].value.isNull())
            m_smallIntCache[i].value = String::number(i);
        return m_smallIntCache[i];
    }

    std::array<StringWithJSString, cacheSize> m_smallIntCache { };
    std::array<CacheEntryWithJSString<int>, cacheSize> m_intCache { };
    std::array<CacheEntryWithJSString<double>, cacheSize> m_doubleCache { };
    std::array<CacheEntry<unsigned>, cacheSize> m_unsignedCache { };
};

} // namespace JSC
