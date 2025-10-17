/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
#include "URLPatternCanonical.h"

#include "URLPattern.h"
#include <wtf/URL.h>
#include <wtf/URLParser.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

static constexpr auto dummyURLCharacters { "https://www.webkit.org"_s };

static bool isValidIPv6HostCodePoint(auto codepoint)
{
    static constexpr std::array validSpecialCodepoints { '[', ']', ':' };
    return isASCIIHexDigit(codepoint) || std::find(validSpecialCodepoints.begin(), validSpecialCodepoints.end(), codepoint) != validSpecialCodepoints.end();
}

// https://urlpattern.spec.whatwg.org/#is-an-absolute-pathname
bool isAbsolutePathname(StringView input, BaseURLStringType inputType)
{
    if (input.isEmpty())
        return false;

    if (input[0] == '/')
        return true;

    if (inputType == BaseURLStringType::URL)
        return false;

    if (input.length() < 2)
        return false;

    if (input.startsWith("\\/"_s))
        return true;

    if (input.startsWith("{/"_s))
        return true;

    return false;
}

// https://urlpattern.spec.whatwg.org/#canonicalize-a-protocol, combined with https://urlpattern.spec.whatwg.org/#process-protocol-for-init
ExceptionOr<String> canonicalizeProtocol(StringView value, BaseURLStringType valueType)
{
    if (value.isEmpty())
        return value.toString();

    auto strippedValue = value.endsWith(':') ? value.left(value.length() - 1) : value;

    if (valueType == BaseURLStringType::Pattern)
        return strippedValue.toString();

    URL dummyURL(makeString(strippedValue, "://webkit.test"_s));

    if (!dummyURL.isValid())
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL protocol string."_s };

    return dummyURL.protocol().toString();
}

// https://urlpattern.spec.whatwg.org/#canonicalize-a-username, combined with https://urlpattern.spec.whatwg.org/#process-username-for-init
String canonicalizeUsername(StringView value, BaseURLStringType valueType)
{
    if (value.isEmpty())
        return value.toString();

    if (valueType == BaseURLStringType::Pattern)
        return value.toString();

    URL dummyURL(dummyURLCharacters);
    dummyURL.setUser(value);

    return dummyURL.encodedUser().toString();
}

// https://urlpattern.spec.whatwg.org/#canonicalize-a-password, combined with https://urlpattern.spec.whatwg.org/#process-password-for-init
String canonicalizePassword(StringView value, BaseURLStringType valueType)
{
    if (value.isEmpty())
        return value.toString();

    if (valueType == BaseURLStringType::Pattern)
        return value.toString();

    URL dummyURL(dummyURLCharacters);
    dummyURL.setPassword(value);

    return dummyURL.encodedPassword().toString();
}

// https://urlpattern.spec.whatwg.org/#canonicalize-a-hostname, combined with https://urlpattern.spec.whatwg.org/#process-hostname-for-init
ExceptionOr<String> canonicalizeHostname(StringView value, BaseURLStringType valueType)
{
    if (value.isEmpty())
        return value.toString();

    if (valueType == BaseURLStringType::Pattern)
        return value.toString();

    // URL::setHost is not fully validating forbidden host code points, so we do it before, except for IPv6 addresses.
    if (value[0] == '[') {
        if (value[value.length() - 1] != ']')
            return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL host string - bad IPv6."_s };
    } else if (value[0] != '[' && value.find(WTF::isForbiddenHostCodePoint) != notFound)
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL host string - forbidden code point."_s };

    URL dummyURL(dummyURLCharacters);
    dummyURL.setHost(value);

    if (!dummyURL.isValid())
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL host string."_s };

    return dummyURL.host().toString();
}

// https://urlpattern.spec.whatwg.org/#canonicalize-an-ipv6-hostname
ExceptionOr<String> canonicalizeIPv6Hostname(StringView value, BaseURLStringType valueType)
{
    if (valueType == BaseURLStringType::Pattern)
        return value.toString();

    StringBuilder result;
    result.reserveCapacity(result.length());

    for (auto codepoint : value.codePoints()) {
        if (!isValidIPv6HostCodePoint(codepoint))
            return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL IPv6 host string."_s };

        result.append(toASCIILower(codepoint));
    }

    return result.toString();
}

// https://urlpattern.spec.whatwg.org/#canonicalize-a-port, combined with https://urlpattern.spec.whatwg.org/#process-port-for-init
ExceptionOr<String> canonicalizePort(StringView portValue, const std::optional<StringView> protocolValue, BaseURLStringType portValueType)
{
    if (portValue.isEmpty())
        return portValue.toString();

    if (portValueType == BaseURLStringType::Pattern)
        return portValue.toString();

    auto parsedPort = parseInteger<uint16_t>(portValue, 10, WTF::ParseIntegerWhitespacePolicy::Disallow);
    if (!parsedPort)
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL port string."_s };

    URL dummyURL(dummyURLCharacters);

    if (protocolValue) {
        if (isDefaultPortForProtocol(*parsedPort, *protocolValue))
            return String { emptyString() };

        dummyURL.setProtocol(*protocolValue);
    }

    dummyURL.setPort(*parsedPort);

    if (!dummyURL.isValid())
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL port string."_s };

    return String::number(*dummyURL.port());
}

// https://urlpattern.spec.whatwg.org/#canonicalize-an-opaque-pathname
ExceptionOr<String> canonicalizeOpaquePathname(StringView value)
{
    if (value.isEmpty())
        return value.toString();

    URL dummyURL(makeString("a:"_s, value));

    if (!dummyURL.isValid())
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL opaque path string."_s };

    return dummyURL.path().toString();
}

// https://urlpattern.spec.whatwg.org/#canonicalize-a-pathname
ExceptionOr<String> canonicalizePathname(StringView pathnameValue)
{
    if (pathnameValue.isEmpty())
        return pathnameValue.toString();

    bool hasLeadingSlash = pathnameValue[0] == '/';
    String maybeAddSlashPrefix = hasLeadingSlash ? pathnameValue.toString() : makeString("/-"_s, pathnameValue);

    // FIXME: Set state override to State::PathStart after URLParser supports state override.
    URL dummyURL(dummyURLCharacters);
    dummyURL.setPath(maybeAddSlashPrefix);

    if (!dummyURL.isValid())
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL path string."_s };

    auto result = dummyURL.path();
    if (!hasLeadingSlash)
        result = result.substring(2);

    return result.toString();
}

// https://urlpattern.spec.whatwg.org/#process-pathname-for-init
ExceptionOr<String> processPathname(StringView pathnameValue, const StringView protocolValue, BaseURLStringType pathnameValueType)
{
    if (pathnameValue.isEmpty())
        return pathnameValue.toString();

    if (pathnameValueType == BaseURLStringType::Pattern)
        return pathnameValue.toString();

    if (WTF::URLParser::isSpecialScheme(protocolValue) || protocolValue.isEmpty())
        return canonicalizePathname(pathnameValue);

    return canonicalizeOpaquePathname(pathnameValue);
}

// https://urlpattern.spec.whatwg.org/#canonicalize-a-search, combined with https://urlpattern.spec.whatwg.org/#process-search-for-init
ExceptionOr<String> canonicalizeSearch(StringView value, BaseURLStringType valueType)
{
    if (value.isEmpty())
        return value.toString();

    auto strippedValue = value[0] == '?' ? value.substring(1) : value;

    if (valueType == BaseURLStringType::Pattern)
        return strippedValue.toString();

    URL dummyURL(dummyURLCharacters);
    dummyURL.setQuery(strippedValue);

    if (!dummyURL.isValid())
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL search string."_s };

    return dummyURL.query().toString();
}

// https://urlpattern.spec.whatwg.org/#canonicalize-a-hash, combined with https://urlpattern.spec.whatwg.org/#process-hash-for-init
ExceptionOr<String> canonicalizeHash(StringView value, BaseURLStringType valueType)
{
    if (value.isEmpty())
        return value.toString();

    auto strippedValue = value[0] == '#' ? value.substring(1) : value;

    if (valueType == BaseURLStringType::Pattern)
        return strippedValue.toString();

    URL dummyURL(dummyURLCharacters);
    dummyURL.setFragmentIdentifier(strippedValue);

    if (!dummyURL.isValid())
        return Exception { ExceptionCode::TypeError, "Invalid input to canonicalize a URL hash string."_s };

    return dummyURL.fragmentIdentifier().toString();
}

ExceptionOr<String> callEncodingCallback(EncodingCallbackType type, StringView input)
{
    switch (type) {
    case EncodingCallbackType::Protocol:
        return canonicalizeProtocol(input, BaseURLStringType::URL);
    case EncodingCallbackType::Username:
        return canonicalizeUsername(input, BaseURLStringType::URL);
    case EncodingCallbackType::Password:
        return canonicalizePassword(input, BaseURLStringType::URL);
    case EncodingCallbackType::Host:
        return canonicalizeHostname(input, BaseURLStringType::URL);
    case EncodingCallbackType::IPv6Host:
        return canonicalizeIPv6Hostname(input, BaseURLStringType::URL);
    case EncodingCallbackType::Port:
        return canonicalizePort(input, { }, BaseURLStringType::URL);
    case EncodingCallbackType::Path:
        return canonicalizePathname(input);
    case EncodingCallbackType::OpaquePath:
        return canonicalizeOpaquePathname(input);
    case EncodingCallbackType::Search:
        return canonicalizeSearch(input, BaseURLStringType::URL);
    case EncodingCallbackType::Hash:
        return canonicalizeHash(input, BaseURLStringType::URL);
    default:
        return Exception { ExceptionCode::TypeError, "Invalid input type for encoding callback."_s };
    }
}

}
