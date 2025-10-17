/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
#include "URLDecomposition.h"

#include "SecurityOrigin.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

String URLDecomposition::origin() const
{
    return SecurityOrigin::create(fullURL())->toString();
}

String URLDecomposition::protocol() const
{
    auto fullURL = this->fullURL();
    if (WTF::protocolIsJavaScript(fullURL.string()))
        return "javascript:"_s;
    return makeString(fullURL.protocol(), ':');
}

void URLDecomposition::setProtocol(StringView value)
{
    URL copy = fullURL();
    copy.setProtocol(value);
    setFullURL(copy);
}

String URLDecomposition::username() const
{
    return fullURL().encodedUser().toString();
}

void URLDecomposition::setUsername(StringView user)
{
    auto fullURL = this->fullURL();
    if (fullURL.host().isEmpty() || fullURL.protocolIsFile())
        return;
    fullURL.setUser(user);
    setFullURL(fullURL);
}

String URLDecomposition::password() const
{
    return fullURL().encodedPassword().toString();
}

void URLDecomposition::setPassword(StringView password)
{
    auto fullURL = this->fullURL();
    if (fullURL.host().isEmpty() || fullURL.protocolIsFile())
        return;
    fullURL.setPassword(password);
    setFullURL(fullURL);
}

String URLDecomposition::host() const
{
    return fullURL().hostAndPort();
}

void URLDecomposition::setHost(StringView value)
{
    auto fullURL = this->fullURL();
    if (value.isEmpty() && !fullURL.protocolIsFile() && fullURL.hasSpecialScheme())
        return;

    if (fullURL.hasOpaquePath())
        return;

    fullURL.setHostAndPort(value);

    if (fullURL.isValid())
        setFullURL(fullURL);
}

String URLDecomposition::hostname() const
{
    return fullURL().host().toString();
}

void URLDecomposition::setHostname(StringView host)
{
    auto fullURL = this->fullURL();
    if (host.isEmpty() && !fullURL.protocolIsFile() && fullURL.hasSpecialScheme())
        return;
    if (fullURL.hasOpaquePath())
        return;
    fullURL.setHost(host);
    if (fullURL.isValid())
        setFullURL(fullURL);
}

String URLDecomposition::port() const
{
    auto port = fullURL().port();
    if (!port)
        return emptyString();
    return String::number(*port);
}

// Outer optional is whether we could parse at all. Inner optional is "no port specified".
static std::optional<std::optional<uint16_t>> parsePort(StringView string, StringView protocol)
{
    // https://url.spec.whatwg.org/#port-state with state override given.
    uint32_t port { 0 };
    bool foundDigit = false;
    for (size_t i = 0; i < string.length(); ++i) {
        auto c = string[i];
        // https://infra.spec.whatwg.org/#ascii-tab-or-newline
        if (c == 0x0009 || c == 0x000A || c == 0x000D)
            continue;
        if (isASCIIDigit(c)) {
            port = port * 10 + c - '0';
            foundDigit = true;
            if (port > std::numeric_limits<uint16_t>::max())
                return std::nullopt;
            continue;
        }
        if (!foundDigit)
            return std::nullopt;
        break;
    }
    if (!foundDigit || WTF::isDefaultPortForProtocol(static_cast<uint16_t>(port), protocol))
        return std::optional<uint16_t> { std::nullopt };
    return {{ static_cast<uint16_t>(port) }};
}

void URLDecomposition::setPort(StringView value)
{
    auto fullURL = this->fullURL();
    if (fullURL.host().isEmpty() || fullURL.protocolIsFile())
        return;
    auto port = parsePort(value, fullURL.protocol());
    if (!port)
        return;
    fullURL.setPort(*port);
    setFullURL(fullURL);
}

String URLDecomposition::pathname() const
{
    return fullURL().path().toString();
}

void URLDecomposition::setPathname(StringView value)
{
    auto fullURL = this->fullURL();
    if (fullURL.hasOpaquePath())
        return;
    fullURL.setPath(value);
    setFullURL(fullURL);
}

String URLDecomposition::search() const
{
    auto fullURL = this->fullURL();
    return fullURL.query().isEmpty() ? emptyString() : fullURL.queryWithLeadingQuestionMark().toString();
}

void URLDecomposition::setSearch(const String& value)
{
    auto fullURL = this->fullURL();
    if (value.isEmpty()) {
        // If the given value is the empty string, set url's query to null.
        fullURL.setQuery({ });
    } else {
        // Make sure that '#' in the query does not leak to the hash.
        fullURL.setQuery(makeStringByReplacingAll(value, '#', "%23"_s));
    }
    setFullURL(fullURL);
}

String URLDecomposition::hash() const
{
    auto fullURL = this->fullURL();
    return fullURL.fragmentIdentifier().isEmpty() ? emptyString() : fullURL.fragmentIdentifierWithLeadingNumberSign().toString();
}

void URLDecomposition::setHash(StringView value)
{
    auto fullURL = this->fullURL();
    if (value.isEmpty())
        fullURL.removeFragmentIdentifier();
    else
        fullURL.setFragmentIdentifier(value.startsWith('#') ? value.substring(1) : value);
    setFullURL(fullURL);
}

}
