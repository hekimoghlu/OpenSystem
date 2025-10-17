/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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
#include "MIMEHeader.h"

#if ENABLE(MHTML)

#include "ParsedContentType.h"
#include "SharedBufferChunkReader.h"
#include <wtf/HashMap.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

typedef HashMap<String, String> KeyValueMap;

static KeyValueMap retrieveKeyValuePairs(WebCore::SharedBufferChunkReader& buffer)
{
    KeyValueMap keyValuePairs;
    String line;
    String key;
    StringBuilder value;
    while (!(line = buffer.nextChunkAsUTF8StringWithLatin1Fallback()).isNull()) {
        if (line.isEmpty())
            break; // Empty line means end of key/value section.
        if (line[0] == '\t') {
            ASSERT(!key.isEmpty());
            value.append(StringView(line).substring(1));
            continue;
        }
        // New key/value, store the previous one if any.
        if (!key.isEmpty()) {
            if (keyValuePairs.find(key) != keyValuePairs.end())
                LOG_ERROR("Key duplicate found in MIME header. Key is '%s', previous value replaced.", key.ascii().data());
            keyValuePairs.add(key, value.toString().trim(deprecatedIsSpaceOrNewline));
            key = String();
            value.clear();
        }
        size_t semicolonIndex = line.find(':');
        if (semicolonIndex == notFound) {
            // This is not a key value pair, ignore.
            continue;
        }
        key = StringView(line).left(semicolonIndex).trim(isUnicodeCompatibleASCIIWhitespace<UChar>).convertToASCIILowercase();
        value.append(StringView(line).substring(semicolonIndex + 1));
    }
    // Store the last property if there is one.
    if (!key.isEmpty())
        keyValuePairs.set(key, value.toString().trim(deprecatedIsSpaceOrNewline));
    return keyValuePairs;
}

RefPtr<MIMEHeader> MIMEHeader::parseHeader(SharedBufferChunkReader& buffer)
{
    auto mimeHeader = adoptRef(*new MIMEHeader);
    KeyValueMap keyValuePairs = retrieveKeyValuePairs(buffer);
    KeyValueMap::iterator mimeParametersIterator = keyValuePairs.find("content-type"_s);
    if (mimeParametersIterator != keyValuePairs.end()) {
        String contentType, charset, multipartType, endOfPartBoundary;
        if (auto parsedContentType = ParsedContentType::create(mimeParametersIterator->value)) {
            contentType = parsedContentType->mimeType();
            charset = parsedContentType->charset().trim(deprecatedIsSpaceOrNewline);
            multipartType = parsedContentType->parameterValueForName("type"_s);
            endOfPartBoundary = parsedContentType->parameterValueForName("boundary"_s);
        }
        mimeHeader->m_contentType = contentType;
        if (!mimeHeader->isMultipart())
            mimeHeader->m_charset = charset;
        else {
            mimeHeader->m_multipartType = multipartType;
            mimeHeader->m_endOfPartBoundary = endOfPartBoundary;
            if (mimeHeader->m_endOfPartBoundary.isNull()) {
                LOG_ERROR("No boundary found in multipart MIME header.");
                return nullptr;
            }
            mimeHeader->m_endOfPartBoundary = makeString("--"_s, mimeHeader->m_endOfPartBoundary);
            mimeHeader->m_endOfDocumentBoundary = makeString(mimeHeader->m_endOfPartBoundary, "--"_s);
        }
    }

    mimeParametersIterator = keyValuePairs.find("content-transfer-encoding"_s);
    if (mimeParametersIterator != keyValuePairs.end())
        mimeHeader->m_contentTransferEncoding = parseContentTransferEncoding(mimeParametersIterator->value);

    mimeParametersIterator = keyValuePairs.find("content-location"_s);
    if (mimeParametersIterator != keyValuePairs.end())
        mimeHeader->m_contentLocation = mimeParametersIterator->value;

    return mimeHeader;
}

MIMEHeader::Encoding MIMEHeader::parseContentTransferEncoding(StringView text)
{
    auto encoding = text.trim(isUnicodeCompatibleASCIIWhitespace<UChar>);
    if (equalLettersIgnoringASCIICase(encoding, "base64"_s))
        return Base64;
    if (equalLettersIgnoringASCIICase(encoding, "quoted-printable"_s))
        return QuotedPrintable;
    if (equalLettersIgnoringASCIICase(encoding, "7bit"_s))
        return SevenBit;
    if (equalLettersIgnoringASCIICase(encoding, "binary"_s))
        return Binary;
    LOG_ERROR("Unknown encoding '%s' found in MIME header.", text.utf8().data());
    return Unknown;
}

MIMEHeader::MIMEHeader()
    : m_contentTransferEncoding(Unknown)
{
}

}

#endif
