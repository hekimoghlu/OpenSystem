/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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

#include "HTTPHeaderNames.h"
#include <utility>
#include <wtf/text/WTFString.h>

namespace WebCore {

// FIXME: Not every header fits into a map. Notably, multiple Set-Cookie header fields are needed to set multiple cookies.

class HTTPHeaderMap {
public:
    struct CommonHeader {
        HTTPHeaderName key;
        String value;

        CommonHeader isolatedCopy() const & { return { key , value.isolatedCopy() }; }
        CommonHeader isolatedCopy() && { return { key , WTFMove(value).isolatedCopy() }; }

        friend bool operator==(const CommonHeader&, const CommonHeader&) = default;
    };

    struct UncommonHeader {
        String key;
        String value;

        UncommonHeader isolatedCopy() const & { return { key.isolatedCopy() , value.isolatedCopy() }; }
        UncommonHeader isolatedCopy() && { return { WTFMove(key).isolatedCopy() , WTFMove(value).isolatedCopy() }; }

        friend bool operator==(const UncommonHeader&, const UncommonHeader&) = default;
    };

    typedef Vector<CommonHeader, 0, CrashOnOverflow, 6> CommonHeadersVector;
    typedef Vector<UncommonHeader, 0, CrashOnOverflow, 0> UncommonHeadersVector;

    class HTTPHeaderMapConstIterator {
    public:
        HTTPHeaderMapConstIterator(const HTTPHeaderMap& table, size_t commonHeadersIndex, size_t uncommonHeadersIndex)
            : m_table(table)
            , m_commonHeadersIndex(commonHeadersIndex)
            , m_uncommonHeadersIndex(uncommonHeadersIndex)
        {
            if (m_commonHeadersIndex < m_table.m_commonHeaders.size()) {
                ASSERT(!m_uncommonHeadersIndex);
                updateKeyValue(m_table.m_commonHeaders[m_commonHeadersIndex]);
            } else if (m_uncommonHeadersIndex < m_table.m_uncommonHeaders.size())
                updateKeyValue(m_table.m_uncommonHeaders[m_uncommonHeadersIndex]);
        }

        struct KeyValue {
            String key;
            std::optional<HTTPHeaderName> keyAsHTTPHeaderName;
            String value;
        };
        using difference_type = ptrdiff_t;
        using value_type = KeyValue;
        using pointer = const KeyValue*;
        using reference = const KeyValue&;
        using iterator_category = std::forward_iterator_tag;

        const KeyValue* get() const
        {
            ASSERT(*this != m_table.end());
            return &m_keyValue;
        }
        const KeyValue& operator*() const { return *get(); }
        const KeyValue* operator->() const { return get(); }

        HTTPHeaderMapConstIterator& operator++()
        {
            if (m_commonHeadersIndex < m_table.m_commonHeaders.size()) {
                ASSERT(!m_uncommonHeadersIndex);
                if (++m_commonHeadersIndex < m_table.m_commonHeaders.size()) {
                    updateKeyValue(m_table.m_commonHeaders[m_commonHeadersIndex]);
                    return *this;
                }
            } else
                ++m_uncommonHeadersIndex;

            if (m_uncommonHeadersIndex < m_table.m_uncommonHeaders.size())
                updateKeyValue(m_table.m_uncommonHeaders[m_uncommonHeadersIndex]);
            return *this;
        }

        bool operator==(const HTTPHeaderMapConstIterator& other) const
        {
            return m_commonHeadersIndex == other.m_commonHeadersIndex && m_uncommonHeadersIndex == other.m_uncommonHeadersIndex;
        }

    private:
        void updateKeyValue(const CommonHeader& header)
        {
            m_keyValue.key = httpHeaderNameString(header.key);
            m_keyValue.keyAsHTTPHeaderName = header.key;
            m_keyValue.value = header.value;
        }

        void updateKeyValue(const UncommonHeader& header)
        {
            m_keyValue.key = header.key;
            m_keyValue.keyAsHTTPHeaderName = std::nullopt;
            m_keyValue.value = header.value;
        }

        const HTTPHeaderMap& m_table;
        size_t m_commonHeadersIndex;
        size_t m_uncommonHeadersIndex;
        KeyValue m_keyValue;
    };
    typedef HTTPHeaderMapConstIterator const_iterator;

    WEBCORE_EXPORT HTTPHeaderMap();
    WEBCORE_EXPORT HTTPHeaderMap(CommonHeadersVector&&, UncommonHeadersVector&&);

    // Gets a copy of the data suitable for passing to another thread.
    WEBCORE_EXPORT HTTPHeaderMap isolatedCopy() const &;
    WEBCORE_EXPORT HTTPHeaderMap isolatedCopy() &&;

    bool isEmpty() const { return m_commonHeaders.isEmpty() && m_uncommonHeaders.isEmpty(); }
    int size() const { return m_commonHeaders.size() + m_uncommonHeaders.size(); }

    void clear()
    {
        m_commonHeaders.clear();
        m_uncommonHeaders.clear();
    }

    void shrinkToFit()
    {
        m_commonHeaders.shrinkToFit();
        m_uncommonHeaders.shrinkToFit();
    }

    WEBCORE_EXPORT String get(StringView name) const;
    WEBCORE_EXPORT void set(const String& name, const String& value);
    WEBCORE_EXPORT void add(const String& name, const String& value);
    void setUncommonHeader(const String& name, const String& value);
    void addUncommonHeader(const String& name, const String& value);
    WEBCORE_EXPORT void append(const String& name, const String& value);
    WEBCORE_EXPORT bool contains(const String&) const;
    WEBCORE_EXPORT bool remove(const String&);

#if USE(CF)
    void set(CFStringRef name, const String& value);
#ifdef __OBJC__
    void set(NSString *name, const String& value) { set((__bridge CFStringRef)name, value); }
#endif
#endif

    WEBCORE_EXPORT String get(HTTPHeaderName) const;
    void set(HTTPHeaderName, const String& value);
    void add(HTTPHeaderName, const String& value);
    bool addIfNotPresent(HTTPHeaderName, const String&);
    WEBCORE_EXPORT bool contains(HTTPHeaderName) const;
    WEBCORE_EXPORT bool remove(HTTPHeaderName);

    // Instead of passing a string literal to any of these functions, just use a HTTPHeaderName instead.
    template<size_t length> String get(ASCIILiteral) const = delete;
    template<size_t length> void set(ASCIILiteral, const String&) = delete;
    template<size_t length> bool contains(ASCIILiteral) = delete;
    template<size_t length> bool remove(ASCIILiteral) = delete;

    const CommonHeadersVector& commonHeaders() const { return m_commonHeaders; }
    const UncommonHeadersVector& uncommonHeaders() const { return m_uncommonHeaders; }
    CommonHeadersVector& commonHeaders() { return m_commonHeaders; }
    UncommonHeadersVector& uncommonHeaders() { return m_uncommonHeaders; }

    const_iterator begin() const { return const_iterator(*this, 0, 0); }
    const_iterator end() const { return const_iterator(*this, m_commonHeaders.size(), m_uncommonHeaders.size()); }

    friend bool operator==(const HTTPHeaderMap& a, const HTTPHeaderMap& b)
    {
        if (a.m_commonHeaders.size() != b.m_commonHeaders.size() || a.m_uncommonHeaders.size() != b.m_uncommonHeaders.size())
            return false;
        for (auto& commonHeader : a.m_commonHeaders) {
            if (b.get(commonHeader.key) != commonHeader.value)
                return false;
        }
        for (auto& uncommonHeader : a.m_uncommonHeaders) {
            if (b.getUncommonHeader(uncommonHeader.key) != uncommonHeader.value)
                return false;
        }
        return true;
    }

private:
    WEBCORE_EXPORT String getUncommonHeader(StringView name) const;

    CommonHeadersVector m_commonHeaders;
    UncommonHeadersVector m_uncommonHeaders;
};

} // namespace WebCore
