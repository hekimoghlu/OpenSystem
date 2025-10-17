/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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

#include <wtf/HashSet.h>
#include <wtf/WallTime.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

typedef UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash> HTTPHeaderSet;

class ResourceResponse;
enum class HTTPHeaderName : uint16_t;

enum class XSSProtectionDisposition {
    Invalid,
    Disabled,
    Enabled,
    BlockEnabled,
};

enum class ContentTypeOptionsDisposition : bool {
    None,
    Nosniff
};

enum class XFrameOptionsDisposition : uint8_t {
    None,
    Deny,
    SameOrigin,
    AllowAll,
    Invalid,
    Conflict
};

enum class CrossOriginResourcePolicy : uint8_t {
    None,
    CrossOrigin,
    SameOrigin,
    SameSite,
    Invalid
};

enum class ClearSiteDataValue : uint8_t {
    Cache = 1 << 0,
    Cookies = 1 << 1,
    ExecutionContexts = 1 << 2,
    Storage = 1 << 3,
};

enum class RangeAllowWhitespace : bool { No, Yes };

bool isValidReasonPhrase(const String&);
bool isValidHTTPHeaderValue(const String&);
bool isValidAcceptHeaderValue(const String&);
bool isValidLanguageHeaderValue(const String&);
#if USE(GLIB)
WEBCORE_EXPORT bool isValidUserAgentHeaderValue(const String&);
#endif
bool isValidHTTPToken(const String&);
bool isValidHTTPToken(StringView);
std::optional<WallTime> parseHTTPDate(const String&);
StringView filenameFromHTTPContentDisposition(StringView);
WEBCORE_EXPORT String extractMIMETypeFromMediaType(const String&);
WEBCORE_EXPORT StringView extractCharsetFromMediaType(StringView);
XSSProtectionDisposition parseXSSProtectionHeader(const String& header, String& failureReason, unsigned& failurePosition, String& reportURL);
AtomString extractReasonPhraseFromHTTPStatusLine(const String&);
WEBCORE_EXPORT XFrameOptionsDisposition parseXFrameOptionsHeader(StringView);
WEBCORE_EXPORT OptionSet<ClearSiteDataValue> parseClearSiteDataHeader(const ResourceResponse&);

// -1 could be set to one of the return parameters to indicate the value is not specified.
WEBCORE_EXPORT bool parseRange(StringView, RangeAllowWhitespace, long long& rangeStart, long long& rangeEnd);

ContentTypeOptionsDisposition parseContentTypeOptionsHeader(StringView header);

// Parsing Complete HTTP Messages.
size_t parseHTTPHeader(std::span<const uint8_t> data, String& failureReason, StringView& nameStr, String& valueStr, bool strict = true);
size_t parseHTTPRequestBody(std::span<const uint8_t> data, Vector<uint8_t>& body);

std::optional<uint64_t> parseContentLength(StringView);

// HTTP Header routine as per https://fetch.spec.whatwg.org/#terminology-headers
bool isForbiddenHeader(const String& name, StringView value);
bool isForbiddenHeaderName(const String&);
bool isNoCORSSafelistedRequestHeaderName(const String&);
bool isPriviledgedNoCORSRequestHeaderName(const String&);
bool isForbiddenResponseHeaderName(const String&);
bool isForbiddenMethod(StringView);
bool isSimpleHeader(const String& name, const String& value);
bool isCrossOriginSafeHeader(HTTPHeaderName, const HTTPHeaderSet&);
bool isCrossOriginSafeHeader(const String&, const HTTPHeaderSet&);
bool isCrossOriginSafeRequestHeader(HTTPHeaderName, const String&);

String normalizeHTTPMethod(const String&);
bool isSafeMethod(const String&);

WEBCORE_EXPORT CrossOriginResourcePolicy parseCrossOriginResourcePolicyHeader(StringView);

template<class HashType>
bool addToAccessControlAllowList(const String& string, unsigned start, unsigned end, UncheckedKeyHashSet<String, HashType>& set)
{
    StringImpl* stringImpl = string.impl();
    if (!stringImpl)
        return true;

    // Skip white space from start.
    while (start <= end && isASCIIWhitespaceWithoutFF((*stringImpl)[start]))
        ++start;

    // only white space
    if (start > end)
        return true;

    // Skip white space from end.
    while (end && isASCIIWhitespaceWithoutFF((*stringImpl)[end]))
        --end;

    auto token = string.substring(start, end - start + 1);
    if (!isValidHTTPToken(token))
        return false;

    set.add(WTFMove(token));
    return true;
}

template<class HashType = DefaultHash<String>>
std::optional<UncheckedKeyHashSet<String, HashType>> parseAccessControlAllowList(const String& string)
{
    UncheckedKeyHashSet<String, HashType> set;
    unsigned start = 0;
    size_t end;
    while ((end = string.find(',', start)) != notFound) {
        if (start != end) {
            if (!addToAccessControlAllowList(string, start, end - 1, set))
                return { };
        }
        start = end + 1;
    }
    if (start != string.length()) {
        if (!addToAccessControlAllowList(string, start, string.length() - 1, set))
            return { };
    }
    return set;
}

}
