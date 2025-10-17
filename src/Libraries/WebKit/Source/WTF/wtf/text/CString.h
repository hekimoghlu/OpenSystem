/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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

#include <span>
#include <wtf/DebugHeap.h>
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(CStringBuffer);

// CStringBuffer is the ref-counted storage class for the characters in a CString.
// The data is implicitly allocated 1 character longer than length(), as it is zero-terminated.
class CStringBuffer final : public RefCounted<CStringBuffer> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(CStringBuffer);
public:
    size_t length() const { return m_length; }

    std::span<const char> span() const LIFETIME_BOUND { return unsafeMakeSpan(m_data, m_length); }
    std::span<const char> unsafeSpanIncludingNullTerminator() const LIFETIME_BOUND { return unsafeMakeSpan(m_data, m_length + 1); }

private:
    friend class CString;

    static Ref<CStringBuffer> createUninitialized(size_t length);

    CStringBuffer(size_t length) : m_length(length) { }
    std::span<char> mutableSpan() LIFETIME_BOUND { return unsafeMakeSpan(m_data, m_length); }
    std::span<char> mutableSpanIncludingNullTerminator() LIFETIME_BOUND { return unsafeMakeSpan(m_data, m_length + 1); }

    const size_t m_length;
    char m_data[0];
};

// A container for a null-terminated char array supporting copy-on-write assignment.
// The contained char array may be null.
class CString final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    CString() { }
    WTF_EXPORT_PRIVATE CString(const char*);
    WTF_EXPORT_PRIVATE CString(std::span<const char>);
    CString(std::span<const LChar>);
    CString(std::span<const char8_t> characters) : CString(byteCast<LChar>(characters)) { }
    CString(CStringBuffer* buffer) : m_buffer(buffer) { }
    WTF_EXPORT_PRIVATE static CString newUninitialized(size_t length, std::span<char>& characterBuffer);
    CString(HashTableDeletedValueType) : m_buffer(HashTableDeletedValue) { }

    const char* data() const LIFETIME_BOUND;

    std::string toStdString() const { return m_buffer ? std::string(m_buffer->unsafeSpanIncludingNullTerminator().data()) : std::string(); }

    std::span<const LChar> span() const LIFETIME_BOUND;
    std::span<const char> unsafeSpanIncludingNullTerminator() const LIFETIME_BOUND;

    WTF_EXPORT_PRIVATE std::span<char> mutableSpan() LIFETIME_BOUND;
    WTF_EXPORT_PRIVATE std::span<char> mutableSpanIncludingNullTerminator() LIFETIME_BOUND;
    size_t length() const;

    bool isNull() const { return !m_buffer; }
    bool isSafeToSendToAnotherThread() const;

    CStringBuffer* buffer() const LIFETIME_BOUND { return m_buffer.get(); }

    bool isHashTableDeletedValue() const { return m_buffer.isHashTableDeletedValue(); }

    WTF_EXPORT_PRIVATE unsigned hash() const;

    // Useful if you want your CString to hold dynamic data.
    WTF_EXPORT_PRIVATE void grow(size_t newLength);

private:
    void copyBufferIfNeeded();
    void init(std::span<const char>);
    RefPtr<CStringBuffer> m_buffer;
};

WTF_EXPORT_PRIVATE bool operator==(const CString&, const CString&);
WTF_EXPORT_PRIVATE bool operator<(const CString&, const CString&);

struct CStringHash {
    static unsigned hash(const CString& string) { return string.hash(); }
    WTF_EXPORT_PRIVATE static bool equal(const CString& a, const CString& b);
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

template<typename> struct DefaultHash;
template<> struct DefaultHash<CString> : CStringHash { };

template<typename> struct HashTraits;
template<> struct HashTraits<CString> : SimpleClassHashTraits<CString> { };

inline CString::CString(std::span<const LChar> bytes)
    : CString(byteCast<char>(bytes))
{
}

inline const char* CString::data() const
{
    return m_buffer ? m_buffer->unsafeSpanIncludingNullTerminator().data() : nullptr;
}

inline std::span<const LChar> CString::span() const
{
    if (m_buffer)
        return byteCast<LChar>(m_buffer->span());
    return { };
}

inline std::span<const char> CString::unsafeSpanIncludingNullTerminator() const
{
    if (m_buffer)
        return m_buffer->unsafeSpanIncludingNullTerminator();
    return { };
}

inline size_t CString::length() const
{
    return m_buffer ? m_buffer->length() : 0;
}

// CString is null terminated
inline const char* safePrintfType(const CString& cstring) { return cstring.data(); }

} // namespace WTF

using WTF::CString;
