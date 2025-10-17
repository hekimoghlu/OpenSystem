/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#include <string>
#include <type_traits>
#include <wtf/ASCIICType.h>
#include <wtf/Compiler.h>
#include <wtf/Forward.h>
#include <wtf/HashFunctions.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/SuperFastHash.h>

#if USE(CF)
typedef const struct __CFString * CFStringRef;
#endif

OBJC_CLASS NSString;

namespace WTF {

class PrintStream;

class ASCIILiteral final {
public:
    constexpr operator const char*() const { return m_charactersWithNullTerminator.data(); }

    static constexpr ASCIILiteral fromLiteralUnsafe(const char* string)
    {
        ASSERT_UNDER_CONSTEXPR_CONTEXT(string);
        return ASCIILiteral { unsafeMakeSpan(string, std::char_traits<char>::length(string) + 1) };
    }

    WTF_EXPORT_PRIVATE void dump(PrintStream& out) const;

    ASCIILiteral() = default;
    constexpr ASCIILiteral(std::nullptr_t)
        : ASCIILiteral()
    { }

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    template<size_t length>
    consteval ASCIILiteral(const char (&literal)[length])
        : m_charactersWithNullTerminator(unsafeMakeSpan(literal, length))
    {
        RELEASE_ASSERT_UNDER_CONSTEXPR_CONTEXT(literal[length - 1] == '\0');
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    unsigned hash() const;
    constexpr bool isNull() const { return m_charactersWithNullTerminator.empty(); }

    constexpr const char* characters() const { return m_charactersWithNullTerminator.data(); }
    constexpr size_t length() const { return !m_charactersWithNullTerminator.empty() ? m_charactersWithNullTerminator.size() - 1 : 0; }
    constexpr std::span<const char> span() const { return m_charactersWithNullTerminator.first(length()); }
    std::span<const LChar> span8() const { return byteCast<LChar>(m_charactersWithNullTerminator.first(length())); }
    std::span<const char> unsafeSpanIncludingNullTerminator() const { return m_charactersWithNullTerminator; }
    size_t isEmpty() const { return m_charactersWithNullTerminator.size() <= 1; }

    constexpr char operator[](size_t index) const { return m_charactersWithNullTerminator[index]; }
    constexpr char characterAt(size_t index) const { return m_charactersWithNullTerminator[index]; }

#ifdef __OBJC__
    // This function convert null strings to empty strings.
    WTF_EXPORT_PRIVATE RetainPtr<NSString> createNSString() const;
#endif

    static ASCIILiteral deletedValue();
    bool isDeletedValue() const { return characters() == reinterpret_cast<char*>(-1); }

#if USE(CF)
    WTF_EXPORT_PRIVATE RetainPtr<CFStringRef> createCFString() const;
#endif

private:
    constexpr explicit ASCIILiteral(std::span<const char> spanWithNullTerminator)
        : m_charactersWithNullTerminator(spanWithNullTerminator)
    {
#if ASSERT_ENABLED
        for (size_t i = 0, size = length(); i < size; ++i)
            ASSERT_UNDER_CONSTEXPR_CONTEXT(isASCII(m_charactersWithNullTerminator[i]));
#endif
    }

    std::span<const char> m_charactersWithNullTerminator;
};

inline bool operator==(ASCIILiteral a, ASCIILiteral b)
{
    if (!a || !b)
        return a.characters() == b.characters();
    return equalSpans(a.span(), b.span());
}

inline unsigned ASCIILiteral::hash() const
{
    if (isNull())
        return 0;
    SuperFastHash hasher;
    hasher.addCharacters(characters(), length());
    return hasher.hash();
}

struct ASCIILiteralHash {
    static unsigned hash(const ASCIILiteral& literal) { return literal.hash(); }
    static bool equal(const ASCIILiteral& a, const ASCIILiteral& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

template<typename T> struct DefaultHash;
template<> struct DefaultHash<ASCIILiteral> : ASCIILiteralHash { };

inline ASCIILiteral ASCIILiteral::deletedValue()
{
    ASCIILiteral result;
    result.m_charactersWithNullTerminator = { reinterpret_cast<char*>(-1), static_cast<size_t>(0) };
    return result;
}

inline namespace StringLiterals {

constexpr ASCIILiteral operator""_s(const char* characters, size_t)
{
    auto result = ASCIILiteral::fromLiteralUnsafe(characters);
    ASSERT_UNDER_CONSTEXPR_CONTEXT(result.characters() == characters);
    return result;
}

constexpr std::span<const char> operator""_span(const char* characters, size_t n)
{
    auto span = unsafeMakeSpan(characters, n);
#if ASSERT_ENABLED
    for (size_t i = 0, size = span.size(); i < size; ++i)
        ASSERT_UNDER_CONSTEXPR_CONTEXT(isASCII(span[i]));
#endif
    return span;
}

constexpr std::span<const LChar> operator""_span8(const char* characters, size_t n)
{
    auto span = byteCast<LChar>(unsafeMakeSpan(characters, n));
#if ASSERT_ENABLED
    for (size_t i = 0, size = span.size(); i < size; ++i)
        ASSERT_UNDER_CONSTEXPR_CONTEXT(isASCII(span[i]));
#endif
    return span;
}

} // inline StringLiterals

// ASCIILiteral is null terminated
inline const char* safePrintfType(const ASCIILiteral& asciiLiteral) { return asciiLiteral.characters(); }

} // namespace WTF

using namespace WTF::StringLiterals;
