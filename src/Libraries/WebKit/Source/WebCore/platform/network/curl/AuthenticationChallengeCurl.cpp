/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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
#include "AuthenticationChallenge.h"

#if USE(CURL)

#include "CurlContext.h"
#include "CurlResponse.h"
#include "ResourceError.h"

namespace WebCore {

AuthenticationChallenge::AuthenticationChallenge(const CurlResponse& curlResponse, unsigned previousFailureCount, const ResourceResponse& response)
    : AuthenticationChallengeBase(protectionSpaceForPasswordBased(curlResponse, response), Credential(), previousFailureCount, response, ResourceError())
{
}

AuthenticationChallenge::AuthenticationChallenge(const URL& url, const CertificateInfo& certificateInfo, const ResourceError& resourceError)
    : AuthenticationChallengeBase(protectionSpaceForServerTrust(url, certificateInfo), Credential(), 0, ResourceResponse(), resourceError)
{
}

ProtectionSpace::ServerType AuthenticationChallenge::protectionSpaceServerTypeFromURI(const URL& url, bool isForProxy)
{
    if (url.protocolIs("https"_s))
        return isForProxy ? ProtectionSpace::ServerType::ProxyHTTPS : ProtectionSpace::ServerType::HTTPS;
    if (url.protocolIs("http"_s))
        return isForProxy ? ProtectionSpace::ServerType::ProxyHTTP : ProtectionSpace::ServerType::HTTP;
    if (url.protocolIs("ftp"_s))
        return isForProxy ? ProtectionSpace::ServerType::ProxyFTP : ProtectionSpace::ServerType::FTP;
    return isForProxy ? ProtectionSpace::ServerType::ProxyHTTP : ProtectionSpace::ServerType::HTTP;
}

ProtectionSpace AuthenticationChallenge::protectionSpaceForPasswordBased(const CurlResponse& curlResponse, const ResourceResponse& response)
{
    if (!response.isUnauthorized() && !response.isProxyAuthenticationRequired())
        return ProtectionSpace();

    auto isProxyAuth = response.isProxyAuthenticationRequired();
    const auto& url = isProxyAuth ? curlResponse.proxyUrl : response.url();
    auto port = determineProxyPort(url);
    auto serverType = protectionSpaceServerTypeFromURI(url, isProxyAuth);
    auto authenticationScheme = authenticationSchemeFromCurlAuth(isProxyAuth ? curlResponse.availableProxyAuth : curlResponse.availableHttpAuth);

    return ProtectionSpace(url.host().toString(), static_cast<int>(port.value_or(0)), serverType, parseRealm(response), authenticationScheme);
}

ProtectionSpace AuthenticationChallenge::protectionSpaceForServerTrust(const URL& url, const CertificateInfo& certificateInfo)
{
    auto port = determineProxyPort(url);
    auto serverType = protectionSpaceServerTypeFromURI(url, false);
    auto authenticationScheme = ProtectionSpace::AuthenticationScheme::ServerTrustEvaluationRequested;

    return ProtectionSpace(url.host().toString(), static_cast<int>(port.value_or(0)), serverType, String(), authenticationScheme, certificateInfo);
}

std::optional<uint16_t> AuthenticationChallenge::determineProxyPort(const URL& url)
{
    static const uint16_t socksPort = 1080;

    if (auto port = url.port())
        return *port;

    if (auto port = defaultPortForProtocol(url.protocol()))
        return *port;

    if (protocolIsInSocksFamily(url))
        return socksPort;

    return std::nullopt;
}

ProtectionSpace::AuthenticationScheme AuthenticationChallenge::authenticationSchemeFromCurlAuth(long curlAuth)
{
    if (curlAuth & CURLAUTH_NTLM)
        return ProtectionSpace::AuthenticationScheme::NTLM;
    if (curlAuth & CURLAUTH_NEGOTIATE)
        return ProtectionSpace::AuthenticationScheme::Negotiate;
    if (curlAuth & CURLAUTH_DIGEST)
        return ProtectionSpace::AuthenticationScheme::HTTPDigest;
    if (curlAuth & CURLAUTH_BASIC)
        return ProtectionSpace::AuthenticationScheme::HTTPBasic;
    return ProtectionSpace::AuthenticationScheme::Unknown;
}

String AuthenticationChallenge::parseRealm(const ResourceResponse& response)
{
    static constexpr auto wwwAuthenticate = "www-authenticate"_s;
    static constexpr auto proxyAuthenticate = "proxy-authenticate"_s;
    static NeverDestroyed<String> realmString(MAKE_STATIC_STRING_IMPL("realm="));

    String realm;
    auto authHeader = response.httpHeaderField(StringView { response.isUnauthorized() ? wwwAuthenticate : proxyAuthenticate });
    auto realmPos = authHeader.findIgnoringASCIICase(realmString.get());
    if (realmPos != notFound) {
        realm = authHeader.substring(realmPos + realmString.get().length());
        realm = realm.left(realm.find(','));
        removeLeadingAndTrailingQuotes(realm);
    }
    return realm;
}

void AuthenticationChallenge::removeLeadingAndTrailingQuotes(String& value)
{
    auto length = value.length();
    if (value.startsWith('"') && value.endsWith('"') && length > 1)
        value = value.substring(1, length - 2);
}

} // namespace WebCore

#endif
