/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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

#include <pal/text/TextEncoding.h>
#include <wtf/RefCounted.h>

namespace PAL {
class TextCodec;
}

namespace WebCore {

class HTMLMetaCharsetParser;

class TextResourceDecoder : public RefCounted<TextResourceDecoder> {
public:
    enum EncodingSource {
        DefaultEncoding,
        AutoDetectedEncoding,
        EncodingFromXMLHeader,
        EncodingFromMetaTag,
        EncodingFromCSSCharset,
        EncodingFromHTTPHeader,
        UserChosenEncoding,
        EncodingFromParentFrame
    };

    WEBCORE_EXPORT static Ref<TextResourceDecoder> create(const String& mimeType, const PAL::TextEncoding& defaultEncoding = { }, bool usesEncodingDetector = false);
    WEBCORE_EXPORT ~TextResourceDecoder();

    static String textFromUTF8(std::span<const uint8_t>);

    void setEncoding(const PAL::TextEncoding&, EncodingSource);
    const PAL::TextEncoding& encoding() const { return m_encoding; }
    const PAL::TextEncoding* encodingForURLParsing();

    bool hasEqualEncodingForCharset(const String& charset) const;

    WEBCORE_EXPORT String decode(std::span<const uint8_t>);
    WEBCORE_EXPORT String flush();
    WEBCORE_EXPORT String decodeAndFlush(std::span<const uint8_t>);

    void setHintEncoding(const TextResourceDecoder* parentFrameDecoder);
   
    void useLenientXMLDecoding() { m_useLenientXMLDecoding = true; }
    bool sawError() const { return m_sawError; }

    void setAlwaysUseUTF8() { ASSERT(m_encoding.name() == "UTF-8"_s); m_alwaysUseUTF8 = true; }

private:
    TextResourceDecoder(const String& mimeType, const PAL::TextEncoding& defaultEncoding, bool usesEncodingDetector);

    enum ContentType { PlainText, HTML, XML, CSS }; // PlainText only checks for BOM.
    static ContentType determineContentType(const String& mimeType);
    static const PAL::TextEncoding& defaultEncoding(ContentType, const PAL::TextEncoding& defaultEncoding);

    size_t checkForBOM(std::span<const uint8_t>);
    bool checkForCSSCharset(std::span<const uint8_t>, bool& movedDataToBuffer);
    bool checkForHeadCharset(std::span<const uint8_t>, bool& movedDataToBuffer);
    bool checkForMetaCharset(std::span<const uint8_t>);
    void detectJapaneseEncoding(std::span<const uint8_t>);
    bool shouldAutoDetect() const;

    ContentType m_contentType;
    PAL::TextEncoding m_encoding;
    std::unique_ptr<PAL::TextCodec> m_codec;
    std::unique_ptr<HTMLMetaCharsetParser> m_charsetParser;
    EncodingSource m_source { DefaultEncoding };
    ASCIILiteral m_parentFrameAutoDetectedEncoding;
    Vector<uint8_t> m_buffer;
    bool m_checkedForBOM { false };
    bool m_checkedForCSSCharset { false };
    bool m_checkedForHeadCharset { false };
    bool m_useLenientXMLDecoding { false }; // Don't stop on XML decoding errors.
    bool m_sawError { false };
    bool m_usesEncodingDetector { false };
    bool m_alwaysUseUTF8 { false };
};

inline void TextResourceDecoder::setHintEncoding(const TextResourceDecoder* parentFrameDecoder)
{
    if (parentFrameDecoder && parentFrameDecoder->m_source == AutoDetectedEncoding)
        m_parentFrameAutoDetectedEncoding = parentFrameDecoder->encoding().name();
}

} // namespace WebCore
