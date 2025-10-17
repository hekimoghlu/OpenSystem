/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#include "ContentSecurityPolicySource.h"

#include "ContentSecurityPolicy.h"
#include "SecurityOriginData.h"
#include <pal/text/TextEncoding.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ContentSecurityPolicySource);

ContentSecurityPolicySource::ContentSecurityPolicySource(const ContentSecurityPolicy& policy, const String& scheme, const String& host, std::optional<uint16_t> port, const String& path, bool hostHasWildcard, bool portHasWildcard, IsSelfSource isSelfSource)
    : m_policy(policy)
    , m_scheme(scheme)
    , m_host(host)
    , m_path(path)
    , m_port(port)
    , m_hostHasWildcard(hostHasWildcard)
    , m_portHasWildcard(portHasWildcard)
    , m_isSelfSource(isSelfSource == IsSelfSource::Yes)
{
}

bool ContentSecurityPolicySource::matches(const URL& url, bool didReceiveRedirectResponse) const
{
    // https://www.w3.org/TR/CSP3/#match-url-to-source-expression.
    if (!schemeMatches(url))
        return false;
    if (isSchemeOnly())
        return true;
    return hostMatches(url) && portMatches(url) && (didReceiveRedirectResponse || pathMatches(url));
}

bool ContentSecurityPolicySource::schemeMatches(const URL& url) const
{
    // https://www.w3.org/TR/CSP3/#match-schemes.
    const auto& scheme = m_scheme.isEmpty() ? m_policy.selfProtocol() : m_scheme;
    auto urlScheme = url.protocol().convertToASCIILowercase();

    if (scheme == urlScheme)
        return true;

    // host-sources can do direct-upgrades.
    if (scheme == "http"_s && urlScheme == "https"_s)
        return true;
    if (scheme == "ws"_s && (urlScheme == "wss"_s || urlScheme == "https"_s || urlScheme == "http"_s))
        return true;
    if (scheme == "wss"_s && urlScheme == "https"_s)
        return true;

    // self-sources can always upgrade to secure protocols and side-grade insecure protocols.
    if ((m_isSelfSource
        && ((urlScheme == "https"_s || urlScheme == "wss"_s) || (scheme == "http"_s && urlScheme == "ws"_s))))
        return true;

    return false;
}

static bool wildcardMatches(StringView host, const String& hostWithWildcard)
{
    auto hostLength = host.length();
    auto hostWithWildcardLength = hostWithWildcard.length();
    return host.endsWithIgnoringASCIICase(hostWithWildcard)
        && hostLength > hostWithWildcardLength
        && host[hostLength - hostWithWildcardLength - 1] == '.';
}

bool ContentSecurityPolicySource::hostMatches(const URL& url) const
{
    auto host = url.host();
    if (m_hostHasWildcard) {
        if (m_host.isEmpty())
            return true;
        return wildcardMatches(host, m_host);
    }
    return equalIgnoringASCIICase(host, m_host);
}

bool ContentSecurityPolicySource::pathMatches(const URL& url) const
{
    if (m_path.isEmpty())
        return true;

    auto path = PAL::decodeURLEscapeSequences(url.path());

    if (m_path.endsWith('/'))
        return path.startsWith(m_path);

    return path == m_path;
}

bool ContentSecurityPolicySource::portMatches(const URL& url) const
{
    if (m_portHasWildcard)
        return true;

    std::optional<uint16_t> port = url.port();

    if (port == m_port)
        return true;

    // host-source and self-source allows upgrading to a more secure scheme which allows for different ports.
    auto defaultSecurePort = WTF::defaultPortForProtocol("https"_s).value_or(443);
    auto defaultInsecurePort = WTF::defaultPortForProtocol("http"_s).value_or(80);
    bool isUpgradeSecure = (port == defaultSecurePort) || (!port && (url.protocol() == "https"_s || url.protocol() == "wss"_s));
    bool isCurrentUpgradable = (m_port == defaultInsecurePort) || (m_scheme == "http"_s && (!m_port || m_port == defaultSecurePort));
    if (isUpgradeSecure && isCurrentUpgradable)
        return true;

    if (!port)
        return WTF::isDefaultPortForProtocol(m_port.value(), url.protocol());

    if (!m_port)
        return WTF::isDefaultPortForProtocol(port.value(), url.protocol());

    return false;
}

bool ContentSecurityPolicySource::isSchemeOnly() const
{
    return m_host.isEmpty() && !m_hostHasWildcard;
}

ContentSecurityPolicySource::operator SecurityOriginData() const
{
    return { m_scheme, m_host, m_port };
}

} // namespace WebCore
