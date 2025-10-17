/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

#if USE(SOUP)

#include "AuthenticationChallenge.h"

#include "ResourceError.h"
#include "URLSoup.h"
#include <libsoup/soup.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

static ProtectionSpace::ServerType protectionSpaceServerTypeFromURL(const URL& url, bool isForProxy)
{
    if (url.protocolIs("https"_s) || url.protocolIs("wss"_s))
        return isForProxy ? ProtectionSpace::ServerType::ProxyHTTPS : ProtectionSpace::ServerType::HTTPS;
    if (url.protocolIs("http"_s) || url.protocolIs("ws"_s))
        return isForProxy ? ProtectionSpace::ServerType::ProxyHTTP : ProtectionSpace::ServerType::HTTP;
    if (url.protocolIs("ftp"_s))
        return isForProxy ? ProtectionSpace::ServerType::ProxyFTP : ProtectionSpace::ServerType::FTP;
    return isForProxy ? ProtectionSpace::ServerType::ProxyHTTP : ProtectionSpace::ServerType::HTTP;
}

static ProtectionSpace protectionSpaceFromSoupAuthAndURL(SoupAuth* soupAuth, const URL& url)
{
    const char* schemeName = soup_auth_get_scheme_name(soupAuth);
    ProtectionSpace::AuthenticationScheme scheme;
    if (!g_ascii_strcasecmp(schemeName, "basic"))
        scheme = ProtectionSpace::AuthenticationScheme::HTTPBasic;
    else if (!g_ascii_strcasecmp(schemeName, "digest"))
        scheme = ProtectionSpace::AuthenticationScheme::HTTPDigest;
    else if (!g_ascii_strcasecmp(schemeName, "ntlm"))
        scheme = ProtectionSpace::AuthenticationScheme::NTLM;
    else if (!g_ascii_strcasecmp(schemeName, "negotiate"))
        scheme = ProtectionSpace::AuthenticationScheme::Negotiate;
    else
        scheme = ProtectionSpace::AuthenticationScheme::Unknown;

#if USE(SOUP2)
    auto host = url.host();
    auto port = url.port();
    if (!port)
        port = defaultPortForProtocol(url.protocol());
#else
    URL authURL({ }, makeString("http://"_s, unsafeSpan(soup_auth_get_authority(soupAuth))));
    auto host = authURL.host();
    auto port = authURL.port();
#endif

    return ProtectionSpace(host.toString(), static_cast<int>(port.value_or(0)),
        protectionSpaceServerTypeFromURL(url, soup_auth_is_for_proxy(soupAuth)),
        String::fromUTF8(soup_auth_get_realm(soupAuth)), scheme);
}

AuthenticationChallenge::AuthenticationChallenge(SoupMessage* soupMessage, SoupAuth* soupAuth, bool retrying)
    : AuthenticationChallengeBase(protectionSpaceFromSoupAuthAndURL(soupAuth, soupURIToURL(soup_message_get_uri(soupMessage)))
        , Credential() // proposedCredentials
        , retrying ? 1 : 0 // previousFailureCount
        , soupMessage // failureResponse
        , ResourceError::authenticationError(soupMessage))
#if USE(SOUP2)
    , m_soupMessage(soupMessage)
#endif
    , m_soupAuth(soupAuth)
{
}

ProtectionSpace AuthenticationChallenge::protectionSpaceForClientCertificate(const URL& url)
{
    auto port = url.port();
    if (!port)
        port = defaultPortForProtocol(url.protocol());
    return ProtectionSpace(url.host().toString(), static_cast<int>(port.value_or(0)), protectionSpaceServerTypeFromURL(url, false), { },
        ProtectionSpace::AuthenticationScheme::ClientCertificateRequested);
}

AuthenticationChallenge::AuthenticationChallenge(SoupMessage* soupMessage, GTlsClientConnection*)
    : AuthenticationChallengeBase(protectionSpaceForClientCertificate(soupURIToURL(soup_message_get_uri(soupMessage)))
        , Credential() // proposedCredentials
        , 0 // previousFailureCount
        , soupMessage // failureResponse
        , ResourceError::authenticationError(soupMessage))
{
}

ProtectionSpace AuthenticationChallenge::protectionSpaceForClientCertificatePassword(const URL& url, GTlsPassword* tlsPassword)
{
    auto port = url.port();
    if (!port)
        port = defaultPortForProtocol(url.protocol());
    return ProtectionSpace(url.host().toString(), static_cast<int>(port.value_or(0)), protectionSpaceServerTypeFromURL(url, false),
        String::fromUTF8(g_tls_password_get_description(tlsPassword)), ProtectionSpace::AuthenticationScheme::ClientCertificatePINRequested);
}

AuthenticationChallenge::AuthenticationChallenge(SoupMessage* soupMessage, GTlsPassword* tlsPassword)
    : AuthenticationChallengeBase(protectionSpaceForClientCertificatePassword(soupURIToURL(soup_message_get_uri(soupMessage)), tlsPassword)
        , Credential() // proposedCredentials
        , g_tls_password_get_flags(tlsPassword) & G_TLS_PASSWORD_RETRY ? 1 : 0 // previousFailureCount
        , soupMessage // failureResponse
        , ResourceError::authenticationError(soupMessage))
    , m_tlsPassword(tlsPassword)
    , m_tlsPasswordFlags(tlsPassword ? g_tls_password_get_flags(tlsPassword) : G_TLS_PASSWORD_NONE)
{
}

bool AuthenticationChallenge::platformCompare(const AuthenticationChallenge& a, const AuthenticationChallenge& b)
{
    if (a.soupAuth() != b.soupAuth())
        return false;

    if (a.tlsPassword() != b.tlsPassword())
        return false;

    if (a.tlsPasswordFlags() != b.tlsPasswordFlags())
        return false;

#if USE(SOUP2)
    return a.soupMessage() == b.soupMessage();
#endif

    return true;
}

} // namespace WebCore

#endif
