/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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

#if ENABLE(MHTML)

#include "MHTMLArchive.h"

#include "Document.h"
#include "LegacySchemeRegistry.h"
#include "LocalFrame.h"
#include "MHTMLParser.h"
#include "MIMETypeRegistry.h"
#include "Page.h"
#include "PageSerializer.h"
#include "QuotedPrintable.h"
#include "SharedBuffer.h"
#include <time.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/DateMath.h>
#include <wtf/GregorianDateTime.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/Base64.h>
#include <wtf/text/StringBuilder.h>

#if HAVE(SYS_TIME_H)
#include <sys/time.h>
#endif

namespace WebCore {

constexpr auto quotedPrintable = "quoted-printable"_s;
constexpr auto base64 = "base64"_s;

static String generateRandomBoundary()
{
    // Trying to generate random boundaries similar to IE/UnMHT (ex: ----=_NextPart_000_001B_01CC157B.96F808A0).
    constexpr size_t randomValuesLength = 10;
    std::array<uint8_t, randomValuesLength> randomValues;
    cryptographicallyRandomValues(randomValues);
    StringBuilder stringBuilder;
    stringBuilder.append("----=_NextPart_000_"_s);
    for (size_t i = 0; i < randomValues.size(); ++i) {
        if (i == 2)
            stringBuilder.append('_');
        else if (i == 6)
            stringBuilder.append('.');
        stringBuilder.append(lowerNibbleToASCIIHexDigit(randomValues[i]));
        stringBuilder.append(upperNibbleToASCIIHexDigit(randomValues[i]));
    }
    return stringBuilder.toString();
}

static String replaceNonPrintableCharacters(const String& text)
{
    StringBuilder stringBuilder;
    for (size_t i = 0; i < text.length(); ++i) {
        if (isASCIIPrintable(text[i]))
            stringBuilder.append(text[i]);
        else
            stringBuilder.append('?');
    }
    return stringBuilder.toString();
}

MHTMLArchive::MHTMLArchive()
{
}

MHTMLArchive::~MHTMLArchive()
{
    // Because all frames know about each other we need to perform a deep clearing of the archives graph.
    clearAllSubframeArchives();
}

Ref<MHTMLArchive> MHTMLArchive::create()
{
    return adoptRef(*new MHTMLArchive);
}

RefPtr<MHTMLArchive> MHTMLArchive::create(const URL& url, FragmentedSharedBuffer& data)
{
    // For security reasons we only load MHTML pages from local URLs.
    if (!LegacySchemeRegistry::shouldTreatURLSchemeAsLocal(url.protocol()))
        return nullptr;

    MHTMLParser parser(&data);
    RefPtr<MHTMLArchive> mainArchive = parser.parseArchive();
    if (!mainArchive)
        return nullptr; // Invalid MHTML file.

    // Since MHTML is a flat format, we need to make all frames aware of all resources.
    for (size_t i = 0; i < parser.frameCount(); ++i) {
        RefPtr<MHTMLArchive> archive = parser.frameAt(i);
        for (size_t j = 1; j < parser.frameCount(); ++j) {
            if (i != j)
                archive->addSubframeArchive(*parser.frameAt(j));
        }
        for (size_t j = 0; j < parser.subResourceCount(); ++j)
            archive->addSubresource(*parser.subResourceAt(j));
    }
    return mainArchive;
}

Ref<FragmentedSharedBuffer> MHTMLArchive::generateMHTMLData(Page* page)
{
    Vector<PageSerializer::Resource> resources;
    PageSerializer pageSerializer(resources);
    pageSerializer.serialize(*page);

    String boundary = generateRandomBoundary();
    String endOfResourceBoundary = makeString("--"_s, boundary, "\r\n"_s);

    GregorianDateTime now;
    now.setToCurrentLocalTime();
    String dateString = makeRFC2822DateString(now.weekDay(), now.monthDay(), now.month(), now.year(), now.hour(), now.minute(), now.second(), now.utcOffsetInMinute());

    StringBuilder stringBuilder;
    stringBuilder.append("From: <Saved by WebKit>\r\n"_s);
    auto* localMainFrame = dynamicDowncast<LocalFrame>(page->mainFrame());
    if (localMainFrame) {
        stringBuilder.append("Subject: "_s);
        // We replace non ASCII characters with '?' characters to match IE's behavior.
        stringBuilder.append(replaceNonPrintableCharacters(localMainFrame->document()->title()));
    }
    stringBuilder.append("\r\nDate: "_s, dateString,
        "\r\nMIME-Version: 1.0\r\nContent-Type: multipart/related;\r\n"_s);
    if (localMainFrame)
        stringBuilder.append("\ttype=\""_s, localMainFrame->document()->suggestedMIMEType());
    stringBuilder.append("\";\r\n\tboundary=\""_s, boundary, "\"\r\n\r\n"_s);

    ASSERT(stringBuilder.toString().containsOnlyASCII());
    CString asciiString = stringBuilder.toString().utf8();
    SharedBufferBuilder mhtmlData;
    mhtmlData.append(asciiString.span());

    for (auto& resource : resources) {
        stringBuilder.clear();
        stringBuilder.append(endOfResourceBoundary, "Content-Type: "_s, resource.mimeType);

        ASCIILiteral contentEncoding;
        if (MIMETypeRegistry::isSupportedJavaScriptMIMEType(resource.mimeType) || MIMETypeRegistry::isSupportedNonImageMIMEType(resource.mimeType))
            contentEncoding = quotedPrintable;
        else
            contentEncoding = base64;

        stringBuilder.append("\r\nContent-Transfer-Encoding: "_s, contentEncoding, "\r\nContent-Location: "_s, resource.url.string(), "\r\n\r\n"_s);

        asciiString = stringBuilder.toString().utf8();
        mhtmlData.append(asciiString.span());

        // FIXME: ideally we would encode the content as a stream without having to fetch it all.
        if (contentEncoding == quotedPrintable) {
            auto encodedData = quotedPrintableEncode(resource.data->span());
            mhtmlData.append(encodedData.span());
            mhtmlData.append("\r\n"_span);
        } else {
            ASSERT(contentEncoding == base64);
            // We are not specifying insertLFs = true below as it would cut the lines with LFs and MHTML requires CRLFs.
            auto encodedData = base64EncodeToVector(resource.data->span());
            const size_t maximumLineLength = 76;
            size_t index = 0;
            size_t encodedDataLength = encodedData.size();
            do {
                size_t lineLength = std::min(encodedDataLength - index, maximumLineLength);
                mhtmlData.append(encodedData.subspan(index, lineLength));
                mhtmlData.append("\r\n"_span);
                index += maximumLineLength;
            } while (index < encodedDataLength);
        }
    }

    asciiString = makeString("--"_s, boundary, "--\r\n"_s).utf8();
    mhtmlData.append(asciiString.span());

    return mhtmlData.take();
}

}

#endif
