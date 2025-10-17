/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
#include "JSStringJoiner.h"

#include "JSCJSValueInlines.h"
#include <wtf/text/ParsingUtilities.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

JSStringJoiner::~JSStringJoiner() = default;

template<typename CharacterType>
static inline void appendStringToData(std::span<CharacterType>& data, StringView string)
{
    if constexpr (std::is_same_v<CharacterType, LChar>) {
        ASSERT(string.is8Bit());
        string.getCharacters8(data);
    } else
        string.getCharacters(data);
    skip(data, string.length());
}

template<typename OutputCharacterType, typename SeparatorCharacterType>
static inline void appendStringToData(std::span<OutputCharacterType>& data, std::span<const SeparatorCharacterType> separator)
{
    StringImpl::copyCharacters(data, separator);
    skip(data, separator.size());
}

template<typename CharacterType>
static inline void appendStringToDataWithOneCharacterSeparatorRepeatedly(std::span<CharacterType>& data, UChar separatorCharacter, StringView string, unsigned count)
{
#if OS(DARWIN)
    if constexpr (std::is_same_v<CharacterType, LChar>) {
        ASSERT(string.is8Bit());
        if (count > 4) {
            switch (string.length() + 1) {
            case 16: {
                alignas(16) LChar pattern[16];
                pattern[0] = separatorCharacter;
                string.getCharacters8(std::span { pattern }.subspan(1));
                size_t fillLength = count * 16;
                memset_pattern16(data.data(), pattern, fillLength);
                skip(data, fillLength);
                return;
            }
            case 8: {
                alignas(8) LChar pattern[8];
                pattern[0] = separatorCharacter;
                string.getCharacters8(std::span { pattern }.subspan(1));
                size_t fillLength = count * 8;
                memset_pattern8(data.data(), pattern, fillLength);
                skip(data, fillLength);
                return;
            }
            case 4: {
                alignas(4) LChar pattern[4];
                pattern[0] = separatorCharacter;
                string.getCharacters8(std::span { pattern }.subspan(1));
                size_t fillLength = count * 4;
                memset_pattern4(data.data(), pattern, fillLength);
                skip(data, fillLength);
                return;
            }
            default:
                break;
            }
        }
    }
#endif

    while (count--) {
        consume(data) = separatorCharacter;
        appendStringToData(data, string);
    }
}

template<typename OutputCharacterType, typename SeparatorCharacterType>
static inline String joinStrings(const JSStringJoiner::Entries& strings, std::span<const SeparatorCharacterType> separator, unsigned joinedLength)
{
    ASSERT(joinedLength);

    std::span<OutputCharacterType> data;
    String result = StringImpl::tryCreateUninitialized(joinedLength, data);
    if (UNLIKELY(result.isNull()))
        return result;

    unsigned size = strings.size();

    switch (separator.size()) {
    case 0: {
        for (unsigned i = 0; i < size; ++i) {
            const auto& entry = strings[i];
            unsigned count = entry.m_additional;
            do {
                appendStringToData(data, entry.m_view.view);
            } while (count--);
        }
        break;
    }
    case 1: {
        OutputCharacterType separatorCharacter = separator.data()[0];
        {
            const auto& entry = strings[0];
            unsigned count = entry.m_additional;
            appendStringToData(data, entry.m_view.view);
            appendStringToDataWithOneCharacterSeparatorRepeatedly(data, separatorCharacter, entry.m_view.view, count);
        }
        for (unsigned i = 1; i < size; ++i) {
            const auto& entry = strings[i];
            unsigned count = entry.m_additional;
            appendStringToDataWithOneCharacterSeparatorRepeatedly(data, separatorCharacter, entry.m_view.view, count + 1);
        }
        break;
    }
    default: {
        {
            const auto& entry = strings[0];
            unsigned count = entry.m_additional;
            appendStringToData(data, entry.m_view.view);
            while (count--) {
                appendStringToData(data, separator);
                appendStringToData(data, entry.m_view.view);
            }
        }
        for (unsigned i = 1; i < size; ++i) {
            const auto& entry = strings[i];
            unsigned count = entry.m_additional;
            do {
                appendStringToData(data, separator);
                appendStringToData(data, entry.m_view.view);
            } while (count--);
        }
        break;
    }
    }
    ASSERT(data.data() == result.span<OutputCharacterType>().data() + joinedLength);

    return result;
}

inline unsigned JSStringJoiner::joinedLength(JSGlobalObject* globalObject) const
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (!m_stringsCount)
        return 0;

    CheckedInt32 separatorLength = m_separator.length();
    CheckedInt32 totalSeparatorsLength = separatorLength * (m_stringsCount - 1);
    CheckedInt32 totalLength = totalSeparatorsLength + m_accumulatedStringsLength;
    if (totalLength.hasOverflowed()) {
        throwOutOfMemoryError(globalObject, scope);
        return 0;
    }
    return totalLength;
}

JSValue JSStringJoiner::joinSlow(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(m_hasOverflowed)) {
        throwOutOfMemoryError(globalObject, scope);
        return { };
    }
    ASSERT(m_strings.size() <= m_strings.capacity());

    unsigned length = joinedLength(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    if (!length)
        return jsEmptyString(vm);

    String result;
    if (m_isAll8Bit)
        result = joinStrings<LChar>(m_strings, m_separator.span8(), length);
    else {
        if (m_separator.is8Bit())
            result = joinStrings<UChar>(m_strings, m_separator.span8(), length);
        else
            result = joinStrings<UChar>(m_strings, m_separator.span16(), length);
    }

    if (UNLIKELY(result.isNull())) {
        throwOutOfMemoryError(globalObject, scope);
        return { };
    }

    return jsString(vm, WTFMove(result));
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
