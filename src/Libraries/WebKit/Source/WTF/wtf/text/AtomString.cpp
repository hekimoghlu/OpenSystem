/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
#include <wtf/text/AtomString.h>

#include <mutex>
#include <wtf/Algorithms.h>
#include <wtf/dtoa.h>
#include <wtf/text/IntegerToStringConversion.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/unicode/CharacterNames.h>

namespace WTF {

const StaticAtomString nullAtomData { nullptr };
const StaticAtomString emptyAtomData { &StringImpl::s_emptyAtomString };

template<AtomString::CaseConvertType type>
ALWAYS_INLINE AtomString AtomString::convertASCIICase() const
{
    StringImpl* impl = this->impl();
    if (UNLIKELY(!impl))
        return nullAtom();

    // Convert short strings without allocating a new StringImpl, since
    // there's a good chance these strings are already in the atom
    // string table and so no memory allocation will be required.
    unsigned length;
    const unsigned localBufferSize = 100;
    if (impl->is8Bit() && (length = impl->length()) <= localBufferSize) {
        auto characters = impl->span8();
        unsigned failingIndex;
        for (unsigned i = 0; i < length; ++i) {
            if (type == CaseConvertType::Lower ? UNLIKELY(isASCIIUpper(characters[i])) : LIKELY(isASCIILower(characters[i]))) {
                failingIndex = i;
                goto SlowPath;
            }
        }
        return *this;
SlowPath:
        std::array<LChar, localBufferSize> localBuffer;
        for (unsigned i = 0; i < failingIndex; ++i)
            localBuffer[i] = characters[i];
        for (unsigned i = failingIndex; i < length; ++i)
            localBuffer[i] = type == CaseConvertType::Lower ? toASCIILower(characters[i]) : toASCIIUpper(characters[i]);
        return std::span<const LChar> { localBuffer }.first(length);
    }

    Ref<StringImpl> convertedString = type == CaseConvertType::Lower ? impl->convertToASCIILowercase() : impl->convertToASCIIUppercase();
    if (LIKELY(convertedString.ptr() == impl))
        return *this;

    AtomString result;
    result.m_string = AtomStringImpl::add(convertedString.ptr());
    return result;
}

AtomString AtomString::convertToASCIILowercase() const
{
    return convertASCIICase<CaseConvertType::Lower>();
}

AtomString AtomString::convertToASCIIUppercase() const
{
    return convertASCIICase<CaseConvertType::Upper>();
}

AtomString AtomString::number(int number)
{
    return numberToStringSigned<AtomString>(number);
}

AtomString AtomString::number(unsigned number)
{
    return numberToStringUnsigned<AtomString>(number);
}

AtomString AtomString::number(unsigned long number)
{
    return numberToStringUnsigned<AtomString>(number);
}

AtomString AtomString::number(unsigned long long number)
{
    return numberToStringUnsigned<AtomString>(number);
}

AtomString AtomString::number(float number)
{
    NumberToStringBuffer buffer;
    auto span = numberToStringAndSize(number, buffer);
    return AtomString { byteCast<LChar>(span) };
}

AtomString AtomString::number(double number)
{
    NumberToStringBuffer buffer;
    auto span = numberToStringAndSize(number, buffer);
    return AtomString { byteCast<LChar>(span) };
}

AtomString AtomString::fromUTF8Internal(std::span<const char> characters)
{
    ASSERT(!characters.empty());
    return AtomStringImpl::add(byteCast<char8_t>(characters));
}

#ifndef NDEBUG

void AtomString::show() const
{
    m_string.show();
}

#endif

static inline StringBuilder replaceUnpairedSurrogatesWithReplacementCharacterInternal(StringView view)
{
    // Slow path: https://infra.spec.whatwg.org/#javascript-string-convert
    // Replaces unpaired surrogates with the replacement character.
    StringBuilder result;
    result.reserveCapacity(view.length());
    for (auto codePoint : view.codePoints()) {
        if (U_IS_SURROGATE(codePoint))
            result.append(replacementCharacter);
        else
            result.append(codePoint);
    }
    return result;
}

AtomString replaceUnpairedSurrogatesWithReplacementCharacter(AtomString&& string)
{
    // Fast path for the case where there are no unpaired surrogates.
    if (LIKELY(!hasUnpairedSurrogate(string)))
        return WTFMove(string);
    return replaceUnpairedSurrogatesWithReplacementCharacterInternal(string).toAtomString();
}

String replaceUnpairedSurrogatesWithReplacementCharacter(String&& string)
{
    // Fast path for the case where there are no unpaired surrogates.
    if (LIKELY(!hasUnpairedSurrogate(string)))
        return WTFMove(string);
    return replaceUnpairedSurrogatesWithReplacementCharacterInternal(string).toString();
}

} // namespace WTF
