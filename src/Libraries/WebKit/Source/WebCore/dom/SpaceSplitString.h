/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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

#include <algorithm>
#include <wtf/MainThread.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class SpaceSplitStringData {
    WTF_MAKE_TZONE_ALLOCATED(SpaceSplitStringData);
    WTF_MAKE_NONCOPYABLE(SpaceSplitStringData);
public:
    static RefPtr<SpaceSplitStringData> create(const AtomString&);

    auto begin() const { return std::to_address(tokenArray().begin()); }
    auto end() const { return std::to_address(tokenArray().end()); }
    auto begin() { return std::to_address(tokenArray().begin()); }
    auto end() { return std::to_address(tokenArray().end()); }

    bool contains(const AtomString& string)
    {
        auto tokens = tokenArray();
        return std::ranges::find(tokens, string) != tokens.end();
    }

    bool containsAll(SpaceSplitStringData&);

    unsigned size() const { return m_size; }
    static constexpr ptrdiff_t sizeMemoryOffset() { return OBJECT_OFFSETOF(SpaceSplitStringData, m_size); }

    const AtomString& operator[](unsigned i) { return tokenArray()[i]; }

    void ref()
    {
        ASSERT(isMainThread());
        ASSERT(m_refCount);
        ++m_refCount;
    }

    void deref()
    {
        ASSERT(isMainThread());
        ASSERT(m_refCount);
        unsigned tempRefCount = m_refCount - 1;
        if (!tempRefCount) {
            destroy(this);
            return;
        }
        m_refCount = tempRefCount;
    }

    const AtomString& keyString() const { return m_keyString; }

    static constexpr ptrdiff_t tokensMemoryOffset() { return sizeof(SpaceSplitStringData); }

private:
    static Ref<SpaceSplitStringData> create(const AtomString&, unsigned tokenCount);
    SpaceSplitStringData(const AtomString& string, unsigned size)
        : m_keyString(string)
        , m_refCount(1)
        , m_size(size)
    {
        ASSERT(!string.isEmpty());
        ASSERT_WITH_MESSAGE(m_size, "SpaceSplitStringData should never be empty by definition. There is no difference between empty and null.");
    }

    ~SpaceSplitStringData() = default;
    static void destroy(SpaceSplitStringData*);

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    std::span<AtomString> tokenArray() { return unsafeMakeSpan(reinterpret_cast<AtomString*>(this + 1), m_size); }
    std::span<const AtomString> tokenArray() const { return unsafeMakeSpan(reinterpret_cast<const AtomString*>(this + 1), m_size); }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    AtomString m_keyString;
    unsigned m_refCount;
    unsigned m_size;
};

class SpaceSplitString {
public:
    SpaceSplitString() = default;

    enum class ShouldFoldCase : bool { No, Yes };
    SpaceSplitString(const AtomString&, ShouldFoldCase);

    const AtomString& keyString() const
    {
        if (m_data)
            return m_data->keyString();
        return nullAtom();
    }

    friend bool operator==(const SpaceSplitString&, const SpaceSplitString&) = default;
    void set(const AtomString&, ShouldFoldCase);
    void clear() { m_data = nullptr; }

    bool contains(const AtomString& string) const { return m_data && m_data->contains(string); }
    bool containsAll(const SpaceSplitString& names) const { return !names.m_data || (m_data && m_data->containsAll(*names.m_data)); }

    unsigned size() const { return m_data ? m_data->size() : 0; }
    bool isEmpty() const { return !m_data; }
    const AtomString& operator[](unsigned i) const
    {
        ASSERT_WITH_SECURITY_IMPLICATION(m_data);
        return (*m_data)[i];
    }

    auto begin() const { return m_data ? m_data->begin() : nullptr; }
    auto end() const { return m_data ? m_data->end() : nullptr; }
    auto begin() { return m_data ? m_data->begin() : nullptr; }
    auto end() { return m_data ? m_data->end() : nullptr; }

    static bool spaceSplitStringContainsValue(StringView spaceSplitString, StringView value, ShouldFoldCase);

private:
    RefPtr<SpaceSplitStringData> m_data;
};

inline SpaceSplitString::SpaceSplitString(const AtomString& string, ShouldFoldCase shouldFoldCase)
    : m_data(!string.isEmpty() ? SpaceSplitStringData::create(shouldFoldCase == ShouldFoldCase::Yes ? string.convertToASCIILowercase() : string) : nullptr)
{
}

} // namespace WebCore
