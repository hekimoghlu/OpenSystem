/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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

#include "ExceptionOr.h"
#include "RegistrableDomain.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/WallTime.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class LoginStatusAuthenticationType : uint8_t { WebAuthn, PasswordManager, Unmanaged };

class LoginStatus {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(LoginStatus, WEBCORE_EXPORT);
public:
    static constexpr uint32_t UsernameMaxLength = 64;
    static constexpr Seconds TimeToLiveAuthentication { 30_s };
    static constexpr Seconds TimeToLiveShort { 24_h * 7 };
    static constexpr Seconds TimeToLiveLong { 24_h * 90 };

    enum class CredentialTokenType : bool { LegacyCookie, HTTPStateToken };
    using AuthenticationType = LoginStatusAuthenticationType;

    WEBCORE_EXPORT static ExceptionOr<UniqueRef<LoginStatus>> create(const RegistrableDomain&, const String& username, CredentialTokenType, AuthenticationType);
    WEBCORE_EXPORT static ExceptionOr<UniqueRef<LoginStatus>> create(const RegistrableDomain&, const String& username, CredentialTokenType, AuthenticationType, Seconds timeToLive);
    WEBCORE_EXPORT LoginStatus(const RegistrableDomain&, const String& username, CredentialTokenType, AuthenticationType, WallTime loggedInTime, Seconds timeToLive);

    WEBCORE_EXPORT void setTimeToLive(Seconds);
    WEBCORE_EXPORT bool hasExpired() const;
    WEBCORE_EXPORT WallTime expiry() const;

    const RegistrableDomain& domain() const { return m_domain; }
    const String& username() const { return m_username; }
    CredentialTokenType tokenType() const { return m_tokenType; }
    AuthenticationType authType() const { return m_authType; }
    WallTime loggedInTime() const { return m_loggedInTime; }
    Seconds timeToLive() const { return m_timeToLive; }

private:
    LoginStatus(const RegistrableDomain&, const String& username, CredentialTokenType, AuthenticationType, Seconds timeToLive);

    RegistrableDomain m_domain;
    String m_username;
    CredentialTokenType m_tokenType;
    AuthenticationType m_authType;
    WallTime m_loggedInTime;
    Seconds m_timeToLive;
};

} // namespace WebCore
