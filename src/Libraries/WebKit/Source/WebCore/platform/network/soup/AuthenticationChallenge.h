/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

#include "AuthenticationChallengeBase.h"
#include "AuthenticationClient.h"

typedef struct _GTlsClientConnection GTlsClientConnection;
typedef struct _GTlsPassword GTlsPassword;
typedef struct _SoupAuth SoupAuth;
typedef struct _SoupMessage SoupMessage;

namespace WebCore {

class AuthenticationChallenge final : public AuthenticationChallengeBase {
public:
    AuthenticationChallenge()
    {
    }

    AuthenticationChallenge(const ProtectionSpace& protectionSpace, const Credential& proposedCredential, unsigned previousFailureCount, const ResourceResponse& response, const ResourceError& error)
        : AuthenticationChallengeBase(protectionSpace, proposedCredential, previousFailureCount, response, error)
    {
    }

    AuthenticationChallenge(const ProtectionSpace& protectionSpace, const Credential& proposedCredential, unsigned previousFailureCount, const ResourceResponse& response, const ResourceError& error, uint32_t tlsPasswordFlags)
        : AuthenticationChallengeBase(protectionSpace, proposedCredential, previousFailureCount, response, error)
        , m_tlsPasswordFlags(tlsPasswordFlags)
    {
    }

    AuthenticationChallenge(SoupMessage*, SoupAuth*, bool retrying);
    AuthenticationChallenge(SoupMessage*, GTlsClientConnection*);
    AuthenticationChallenge(SoupMessage*, GTlsPassword*);
    AuthenticationClient* authenticationClient() const { RELEASE_ASSERT_NOT_REACHED(); }
#if USE(SOUP2)
    SoupMessage* soupMessage() const { return m_soupMessage.get(); }
#endif
    SoupAuth* soupAuth() const { return m_soupAuth.get(); }
    GTlsPassword* tlsPassword() const { return m_tlsPassword.get(); }
    void setProposedCredential(const Credential& credential) { m_proposedCredential = credential; }

    uint32_t tlsPasswordFlags() const { return m_tlsPasswordFlags; }
    void setTLSPasswordFlags(uint32_t tlsPasswordFlags) { m_tlsPasswordFlags = tlsPasswordFlags; }

    static ProtectionSpace protectionSpaceForClientCertificate(const URL&);
    static ProtectionSpace protectionSpaceForClientCertificatePassword(const URL&, GTlsPassword*);

private:
    friend class AuthenticationChallengeBase;
    static bool platformCompare(const AuthenticationChallenge&, const AuthenticationChallenge&);

#if USE(SOUP2)
    GRefPtr<SoupMessage> m_soupMessage;
#endif
    GRefPtr<SoupAuth> m_soupAuth;
    GRefPtr<GTlsPassword> m_tlsPassword;
    uint32_t m_tlsPasswordFlags { 0 };
};

} // namespace WebCore

