/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "ProtectionSpaceBase.h"

#include "ProtectionSpace.h"

namespace WebCore {
 
// Need to enforce empty, non-null strings due to the pickiness of the String == String operator
// combined with the semantics of the String(NSString*) constructor
ProtectionSpaceBase::ProtectionSpaceBase(const String& host, int port, ServerType serverType, const String& realm, AuthenticationScheme authenticationScheme)
    : m_host(host.length() ? host : emptyString())
    , m_realm(realm.length() ? realm : emptyString())
    , m_port(port)
    , m_serverType(serverType)
    , m_authenticationScheme(authenticationScheme)
{    
}

bool ProtectionSpaceBase::isProxy() const
{
    return m_serverType == ServerType::ProxyHTTP
        || m_serverType == ServerType::ProxyHTTPS
        || m_serverType == ServerType::ProxyFTP
        || m_serverType == ServerType::ProxySOCKS;
}

bool ProtectionSpaceBase::receivesCredentialSecurely() const
{
    return m_serverType == ServerType::HTTPS
        || m_serverType == ServerType::FTPS
        || m_serverType == ServerType::ProxyHTTPS
        || m_authenticationScheme == AuthenticationScheme::HTTPDigest;
}

bool ProtectionSpaceBase::isPasswordBased() const
{
    switch (m_authenticationScheme) {
    case AuthenticationScheme::Default:
    case AuthenticationScheme::HTTPBasic:
    case AuthenticationScheme::HTTPDigest:
    case AuthenticationScheme::HTMLForm:
    case AuthenticationScheme::NTLM:
    case AuthenticationScheme::Negotiate:
    case AuthenticationScheme::OAuth:
#if PLATFORM(COCOA)
    case AuthenticationScheme::XMobileMeAuthToken:
    case AuthenticationScheme::PrivateAccessToken:
    case AuthenticationScheme::OAuthBearerToken:
#endif
#if USE(GLIB)
    case AuthenticationScheme::ClientCertificatePINRequested:
#endif
        return true;
    case AuthenticationScheme::ClientCertificateRequested:
    case AuthenticationScheme::ServerTrustEvaluationRequested:
#if !PLATFORM(COCOA)
    case AuthenticationScheme::Unknown:
#endif
        return false;
    }

    return true;
}

bool ProtectionSpaceBase::compare(const ProtectionSpace& a, const ProtectionSpace& b)
{
    if (a.host() != b.host())
        return false;
    if (a.port() != b.port())
        return false;
    if (a.serverType() != b.serverType())
        return false;
    // Ignore realm for proxies
    if (!a.isProxy() && a.realm() != b.realm())
        return false;
    if (a.authenticationScheme() != b.authenticationScheme())
        return false;

    return ProtectionSpace::platformCompare(a, b);
}

}
