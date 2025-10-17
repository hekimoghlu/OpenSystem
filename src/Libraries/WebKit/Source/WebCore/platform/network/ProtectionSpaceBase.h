/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

class ProtectionSpace;

enum class ProtectionSpaceBaseServerType : uint8_t {
    HTTP = 1, // NOLINT
    HTTPS, // NOLINT
    FTP, // NOLINT
    FTPS, // NOLINT
    ProxyHTTP,
    ProxyHTTPS,
    ProxyFTP,
    ProxySOCKS
};

enum class ProtectionSpaceBaseAuthenticationScheme : uint8_t {
    Default = 1,
    HTTPBasic,
    HTTPDigest,
    HTMLForm,
    NTLM, // NOLINT
    Negotiate,
    ClientCertificateRequested,
    ServerTrustEvaluationRequested,
#if PLATFORM(COCOA)
    XMobileMeAuthToken,
#endif
    OAuth,
#if PLATFORM(COCOA)
    PrivateAccessToken,
    OAuthBearerToken,
#endif
#if USE(GLIB)
    ClientCertificatePINRequested,
#endif
#if !PLATFORM(COCOA)
    Unknown = 100
#endif
};
  
class ProtectionSpaceBase {
public:
    using ServerType = ProtectionSpaceBaseServerType;
    using AuthenticationScheme = ProtectionSpaceBaseAuthenticationScheme;

    bool isHashTableDeletedValue() const { return m_isHashTableDeletedValue; }
    
    const String& host() const { return m_host; }
    int port() const { return m_port; }
    ServerType serverType() const { return m_serverType; }
    WEBCORE_EXPORT bool isProxy() const;
    const String& realm() const { return m_realm; }
    AuthenticationScheme authenticationScheme() const { return m_authenticationScheme; }
    
    WEBCORE_EXPORT bool receivesCredentialSecurely() const;
    WEBCORE_EXPORT bool isPasswordBased() const;

    bool encodingRequiresPlatformData() const { return false; }

    WEBCORE_EXPORT static bool compare(const ProtectionSpace&, const ProtectionSpace&);

protected:
    ProtectionSpaceBase() = default;
    WEBCORE_EXPORT ProtectionSpaceBase(const String& host, int port, ServerType, const String& realm, AuthenticationScheme);

    // Hash table deleted values, which are only constructed and never copied or destroyed.
    ProtectionSpaceBase(WTF::HashTableDeletedValueType) : m_isHashTableDeletedValue(true) { }

    static bool platformCompare(const ProtectionSpace&, const ProtectionSpace&) { return true; }

private:
    // Need to enforce empty, non-null strings due to the pickiness of the String == String operator
    // combined with the semantics of the String(NSString*) constructor
    String m_host { emptyString() };
    String m_realm { emptyString() };

    int m_port { 0 };
    ServerType m_serverType { ServerType::HTTP };
    AuthenticationScheme m_authenticationScheme { AuthenticationScheme::Default };
    bool m_isHashTableDeletedValue { false };
};

inline bool operator==(const ProtectionSpace& a, const ProtectionSpace& b) { return ProtectionSpaceBase::compare(a, b); }
    
} // namespace WebCore

