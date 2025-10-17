/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

#if USE(CURL)
#include "ResourceResponse.h"

#include "CurlContext.h"
#include "CurlResponse.h"
#include "HTTPParsers.h"

namespace WebCore {

static bool isAppendableHeader(const String &key)
{
    static constexpr ASCIILiteral appendableHeaders[] = {
        "access-control-allow-headers"_s,
        "access-control-allow-methods"_s,
        "access-control-allow-origin"_s,
        "access-control-expose-headers"_s,
        "allow"_s,
        "cache-control"_s,
        "connection"_s,
        "content-encoding"_s,
        "content-language"_s,
        "if-match"_s,
        "if-none-match"_s,
        "keep-alive"_s,
        "pragma"_s,
        "proxy-authenticate"_s,
        "public"_s,
        "server"_s,
        "set-cookie"_s,
        "te"_s,
        "timing-allow-origin"_s,
        "trailer"_s,
        "transfer-encoding"_s,
        "upgrade"_s,
        "user-agent"_s,
        "vary"_s,
        "via"_s,
        "warning"_s,
        "www-authenticate"_s
    };

    // Custom headers start with 'X-', and need no further checking.
    if (startsWithLettersIgnoringASCIICase(key, "x-"_s))
        return true;

    for (const auto& header : appendableHeaders) {
        if (equalIgnoringASCIICase(key, header))
            return true;
    }

    return false;
}

ResourceResponse::ResourceResponse(CurlResponse& response)
    : ResourceResponseBase()
{
    setURL(response.url);
    setExpectedContentLength(response.expectedContentLength);
    setHTTPStatusCode(response.statusCode ? response.statusCode : response.httpConnectCode);

    for (const auto& header : response.headers)
        appendHTTPHeaderField(header);

    switch (response.httpVersion) {
    case CURL_HTTP_VERSION_1_0:
        setHTTPVersion("HTTP/1.0"_s);
        break;
    case CURL_HTTP_VERSION_1_1:
        setHTTPVersion("HTTP/1.1"_s);
        break;
    case CURL_HTTP_VERSION_2_0:
        setHTTPVersion("HTTP/2"_s);
        break;
    case CURL_HTTP_VERSION_3:
        setHTTPVersion("HTTP/3"_s);
        break;
    case CURL_HTTP_VERSION_NONE:
    default:
        break;
    }

    setMimeType(extractMIMETypeFromMediaType(httpHeaderField(HTTPHeaderName::ContentType)).convertToASCIILowercase());
    setTextEncodingName(extractCharsetFromMediaType(httpHeaderField(HTTPHeaderName::ContentType)).toString());
    setCertificateInfo(WTFMove(response.certificateInfo));
    setSource(ResourceResponse::Source::Network);
}

void ResourceResponse::appendHTTPHeaderField(const String& header)
{
    if (startsWithLettersIgnoringASCIICase(header, "http/"_s)) {
        setHTTPStatusText(String { extractReasonPhraseFromHTTPStatusLine(header.trim(deprecatedIsSpaceOrNewline)) });
        return;
    }

    if (auto splitPosition = header.find(':'); splitPosition != notFound) {
        auto key = header.left(splitPosition).trim(deprecatedIsSpaceOrNewline);
        auto value = header.substring(splitPosition + 1).trim(deprecatedIsSpaceOrNewline);

        if (isAppendableHeader(key))
            addHTTPHeaderField(key, value);
        else
            setHTTPHeaderField(key, value);
    }
}

String ResourceResponse::platformSuggestedFilename() const
{
    StringView contentDisposition = filenameFromHTTPContentDisposition(httpHeaderField(HTTPHeaderName::ContentDisposition));
    if (contentDisposition.is8Bit())
        return String::fromUTF8WithLatin1Fallback(contentDisposition.span8());
    return contentDisposition.toString();
}

}

#endif
