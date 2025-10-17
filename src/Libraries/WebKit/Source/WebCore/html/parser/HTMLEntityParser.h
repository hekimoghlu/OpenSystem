/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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
#include <span>
#include <unicode/umachine.h>
#include <wtf/Forward.h>
#include <wtf/text/LChar.h>

namespace WebCore {

class DecodedHTMLEntity;
class SegmentedString;

// This function expects a null character at the end, otherwise it assumes the source is partial.
DecodedHTMLEntity consumeHTMLEntity(SegmentedString&, UChar additionalAllowedCharacter = 0);

// This function assumes the source is complete, and does not expect a null character.
DecodedHTMLEntity consumeHTMLEntity(StringParsingBuffer<LChar>&);
DecodedHTMLEntity consumeHTMLEntity(StringParsingBuffer<UChar>&);

// This function does not check for "not enough characters" at all.
DecodedHTMLEntity decodeNamedHTMLEntityForXMLParser(const char*);

class DecodedHTMLEntity {
public:
    constexpr DecodedHTMLEntity();
    constexpr DecodedHTMLEntity(UChar);
    constexpr DecodedHTMLEntity(UChar, UChar);
    constexpr DecodedHTMLEntity(UChar, UChar, UChar);

    enum ConstructNotEnoughCharactersType { ConstructNotEnoughCharacters };
    constexpr DecodedHTMLEntity(ConstructNotEnoughCharactersType);

    constexpr bool failed() const { return !m_length; }
    constexpr bool notEnoughCharacters() const { return m_notEnoughCharacters; }

    constexpr std::span<const UChar> span() const { return std::span { m_characters }.first(m_length); }

private:
    uint8_t m_length { 0 };
    bool m_notEnoughCharacters { false };
    std::array<UChar, 3> m_characters;
};

} // namespace WebCore
