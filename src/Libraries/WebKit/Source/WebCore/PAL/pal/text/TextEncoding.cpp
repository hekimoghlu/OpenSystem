/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#include "TextEncoding.h"

#include "DecodeEscapeSequences.h"
#include "TextCodec.h"
#include "TextEncodingRegistry.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/StringView.h>

namespace PAL {

static const TextEncoding& UTF7Encoding()
{
    static NeverDestroyed<TextEncoding> globalUTF7Encoding("UTF-7"_s);
    return globalUTF7Encoding;
}

TextEncoding::TextEncoding(ASCIILiteral name)
    : m_name(atomCanonicalTextEncodingName(name))
    , m_backslashAsCurrencySymbol(backslashAsCurrencySymbol())
{
}

TextEncoding::TextEncoding(StringView name)
    : m_name(atomCanonicalTextEncodingName(name))
    , m_backslashAsCurrencySymbol(backslashAsCurrencySymbol())
{
}

TextEncoding::TextEncoding(const String& name)
    : TextEncoding(StringView { name })
{
}

String TextEncoding::decode(std::span<const uint8_t> data, bool stopOnError, bool& sawError) const
{
    if (m_name.isNull())
        return String();

    return newTextCodec(*this)->decode(data, true, stopOnError, sawError);
}

Vector<uint8_t> TextEncoding::encode(StringView string, PAL::UnencodableHandling handling, NFCNormalize normalize) const
{
    if (m_name.isNull() || string.isEmpty())
        return { };

    // FIXME: What's the right place to do normalization?
    // It's a little strange to do it inside the encode function.
    // Perhaps normalization should be an explicit step done before calling encode.
    if (normalize == NFCNormalize::Yes)
        return newTextCodec(*this)->encode(normalizedNFC(string).view, handling);
    return newTextCodec(*this)->encode(string, handling);
}

ASCIILiteral TextEncoding::domName() const
{
    if (noExtendedTextEncodingNameUsed())
        return m_name;

    // We treat EUC-KR as windows-949 (its superset), but need to expose 
    // the name 'EUC-KR' because the name 'windows-949' is not recognized by
    // most Korean web servers even though they do use the encoding
    // 'windows-949' with the name 'EUC-KR'. 
    // FIXME: This is not thread-safe. At the moment, this function is
    // only accessed in a single thread, but eventually has to be made
    // thread-safe along with usesVisualOrdering().
    static const ASCIILiteral windows949 = atomCanonicalTextEncodingName("windows-949"_s);
    if (m_name == windows949)
        return "EUC-KR"_s;
    return m_name;
}

bool TextEncoding::usesVisualOrdering() const
{
    if (noExtendedTextEncodingNameUsed())
        return false;

    static const ASCIILiteral iso88598 = atomCanonicalTextEncodingName("ISO-8859-8"_s);
    return m_name == iso88598;
}

bool TextEncoding::isJapanese() const
{
    return isJapaneseEncoding(m_name);
}

UChar TextEncoding::backslashAsCurrencySymbol() const
{
    return shouldShowBackslashAsCurrencySymbolIn(m_name) ? 0x00A5 : '\\';
}

bool TextEncoding::isNonByteBasedEncoding() const
{
    return *this == UTF16LittleEndianEncoding() || *this == UTF16BigEndianEncoding();
}

bool TextEncoding::isUTF7Encoding() const
{
    if (noExtendedTextEncodingNameUsed())
        return false;

    return *this == UTF7Encoding();
}

const TextEncoding& TextEncoding::closestByteBasedEquivalent() const
{
    if (isNonByteBasedEncoding())
        return UTF8Encoding();
    return *this; 
}

// HTML5 specifies that UTF-8 be used in form submission when a form is 
// is a part of a document in UTF-16 probably because UTF-16 is not a 
// byte-based encoding and can contain 0x00. By extension, the same
// should be done for UTF-32. In case of UTF-7, it is a byte-based encoding,
// but it's fraught with problems and we'd rather steer clear of it.
const TextEncoding& TextEncoding::encodingForFormSubmissionOrURLParsing() const
{
    if (isNonByteBasedEncoding() || isUTF7Encoding())
        return UTF8Encoding();
    return *this;
}

const TextEncoding& ASCIIEncoding()
{
    static NeverDestroyed<TextEncoding> globalASCIIEncoding("ASCII"_s);
    return globalASCIIEncoding;
}

const TextEncoding& Latin1Encoding()
{
    static NeverDestroyed<TextEncoding> globalLatin1Encoding("latin1"_s);
    return globalLatin1Encoding;
}

const TextEncoding& UTF16BigEndianEncoding()
{
    static NeverDestroyed<TextEncoding> globalUTF16BigEndianEncoding("UTF-16BE"_s);
    return globalUTF16BigEndianEncoding;
}

const TextEncoding& UTF16LittleEndianEncoding()
{
    static NeverDestroyed<TextEncoding> globalUTF16LittleEndianEncoding("UTF-16LE"_s);
    return globalUTF16LittleEndianEncoding;
}

const TextEncoding& UTF8Encoding()
{
    static NeverDestroyed<TextEncoding> globalUTF8Encoding("UTF-8"_s);
    ASSERT(globalUTF8Encoding.get().isValid());
    return globalUTF8Encoding;
}

const TextEncoding& WindowsLatin1Encoding()
{
    static NeverDestroyed<TextEncoding> globalWindowsLatin1Encoding("WinLatin-1"_s);
    return globalWindowsLatin1Encoding;
}

String decodeURLEscapeSequences(StringView string, const TextEncoding& encoding)
{
    if (string.isEmpty())
        return string.toString();
    return decodeEscapeSequences<URLEscapeSequence>(string, encoding);
}

} // namespace PAL
