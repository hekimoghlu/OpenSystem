/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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

#include "UnencodableHandling.h"
#include <wtf/URL.h>
#include <wtf/text/StringView.h>

namespace PAL {

enum class NFCNormalize : bool { No, Yes };

class TextEncoding : public WTF::URLTextEncoding {
public:
    TextEncoding() = default;
    PAL_EXPORT TextEncoding(ASCIILiteral name);
    PAL_EXPORT TextEncoding(StringView name);
    PAL_EXPORT TextEncoding(const String& name);

    bool isValid() const { return !m_name.isNull(); }
    ASCIILiteral name() const { return m_name; }
    PAL_EXPORT ASCIILiteral domName() const; // name exposed via DOM
    bool usesVisualOrdering() const;
    bool isJapanese() const;

    const TextEncoding& closestByteBasedEquivalent() const;
    const TextEncoding& encodingForFormSubmissionOrURLParsing() const;

    PAL_EXPORT String decode(std::span<const uint8_t>, bool stopOnError, bool& sawError) const;
    String decode(std::span<const uint8_t>) const;
    PAL_EXPORT Vector<uint8_t> encode(StringView, PAL::UnencodableHandling, NFCNormalize = NFCNormalize::Yes) const;
    Vector<uint8_t> encodeForURLParsing(StringView string) const final { return encode(string, PAL::UnencodableHandling::URLEncodedEntities, NFCNormalize::No); }

    UChar backslashAsCurrencySymbol() const;
    bool isByteBasedEncoding() const { return !isNonByteBasedEncoding(); }

private:
    bool isNonByteBasedEncoding() const;
    bool isUTF7Encoding() const;

    ASCIILiteral m_name;
    UChar m_backslashAsCurrencySymbol;
};

inline bool operator==(const TextEncoding& a, const TextEncoding& b) { return a.name() == b.name(); }

const TextEncoding& ASCIIEncoding();
const TextEncoding& Latin1Encoding();
const TextEncoding& UTF16BigEndianEncoding();
const TextEncoding& UTF16LittleEndianEncoding();
PAL_EXPORT const TextEncoding& UTF8Encoding();
PAL_EXPORT const TextEncoding& WindowsLatin1Encoding();

// Unescapes the given string using URL escaping rules.
// DANGER: If the URL has "%00" in it,
// the resulting string will have embedded null characters!
PAL_EXPORT String decodeURLEscapeSequences(StringView, const TextEncoding& = UTF8Encoding());

inline String TextEncoding::decode(std::span<const uint8_t> characters) const
{
    bool ignored;
    return decode(characters, false, ignored);
}

} // namespace PAL
