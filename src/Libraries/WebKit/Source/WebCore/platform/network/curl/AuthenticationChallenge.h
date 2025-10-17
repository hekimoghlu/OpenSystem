/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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

namespace WebCore {

class CurlResponse;

class AuthenticationChallenge final : public AuthenticationChallengeBase {
public:
    AuthenticationChallenge()
    {
    }

    AuthenticationChallenge(const ProtectionSpace& protectionSpace, const Credential& proposedCredential, unsigned previousFailureCount, const ResourceResponse& response, const ResourceError& error)
        : AuthenticationChallengeBase(protectionSpace, proposedCredential, previousFailureCount, response, error)
    {
    }

    WEBCORE_EXPORT AuthenticationChallenge(const CurlResponse&, unsigned, const ResourceResponse&);
    WEBCORE_EXPORT AuthenticationChallenge(const URL&, const CertificateInfo&, const ResourceError&);

    AuthenticationClient* authenticationClient() const { return nullptr; }

private:
    ProtectionSpace::ServerType protectionSpaceServerTypeFromURI(const URL&, bool isForProxy);
    ProtectionSpace protectionSpaceForPasswordBased(const CurlResponse&, const ResourceResponse&);
    ProtectionSpace protectionSpaceForServerTrust(const URL&, const CertificateInfo&);
    std::optional<uint16_t> determineProxyPort(const URL&);
    ProtectionSpace::AuthenticationScheme authenticationSchemeFromCurlAuth(long);
    String parseRealm(const ResourceResponse&);
    void removeLeadingAndTrailingQuotes(String&);
};

} // namespace WebCore
