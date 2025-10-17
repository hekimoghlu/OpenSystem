/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#include "LoginStatus.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringCommon.h>

namespace WebCore {

using CodeUnitMatchFunction = bool (*)(UChar);

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LoginStatus);

ExceptionOr<UniqueRef<LoginStatus>> LoginStatus::create(const RegistrableDomain& domain, const String& username, CredentialTokenType tokenType, AuthenticationType authType)
{
    return create(domain, username, tokenType, authType, authType == AuthenticationType::Unmanaged ? TimeToLiveShort : TimeToLiveLong);
}

ExceptionOr<UniqueRef<LoginStatus>> LoginStatus::create(const RegistrableDomain& domain, const String& username, CredentialTokenType tokenType, AuthenticationType authType, Seconds timeToLive)
{
    if (domain.isEmpty())
        return Exception { ExceptionCode::SecurityError, "LoginStatus status can only be set for origins with a registrable domain."_s };

    unsigned length = username.length();
    if (length > UsernameMaxLength)
        return Exception { ExceptionCode::SyntaxError, makeString("LoginStatus usernames cannot be longer than "_s, UsernameMaxLength) };

    auto spaceOrNewline = username.find([](UChar ch) {
        return deprecatedIsSpaceOrNewline(ch);
    });
    if (spaceOrNewline != notFound)
        return Exception { ExceptionCode::InvalidCharacterError, "LoginStatus usernames cannot contain whitespace or newlines."_s };

    return makeUniqueRef<LoginStatus>(*new LoginStatus(domain, username, tokenType, authType, timeToLive));
}

LoginStatus::LoginStatus(const RegistrableDomain& domain, const String& username, CredentialTokenType tokenType, AuthenticationType authType, Seconds timeToLive)
    : m_domain { domain }
    , m_username { username }
    , m_tokenType { tokenType }
    , m_authType { authType }
    , m_loggedInTime { WallTime::now() }
{
    setTimeToLive(timeToLive);
}

LoginStatus::LoginStatus(const RegistrableDomain& domain, const String& username, CredentialTokenType tokenType, AuthenticationType authType, WallTime loggedInTime, Seconds timeToLive)
    : m_domain { domain }
    , m_username { username }
    , m_tokenType { tokenType }
    , m_authType { authType }
    , m_loggedInTime { loggedInTime }
{
    setTimeToLive(timeToLive);
}

void LoginStatus::setTimeToLive(Seconds timeToLive)
{
    m_timeToLive = std::min(timeToLive, m_authType == AuthenticationType::Unmanaged ? TimeToLiveShort : TimeToLiveLong);
}

bool LoginStatus::hasExpired() const
{
    ASSERT(!m_domain.isEmpty());
    return WallTime::now() > m_loggedInTime + m_timeToLive;
}

WallTime LoginStatus::expiry() const
{
    return WallTime::now() + m_timeToLive;
}

}
