/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
#include <wtf/RefPtr.h>

OBJC_CLASS NSURLAuthenticationChallenge;

namespace WebCore {

class AuthenticationChallenge : public AuthenticationChallengeBase {
public:
    AuthenticationChallenge() {}
    WEBCORE_EXPORT AuthenticationChallenge(const ProtectionSpace&, const Credential& proposedCredential, unsigned previousFailureCount, const ResourceResponse&, const ResourceError&);

    WEBCORE_EXPORT AuthenticationChallenge(NSURLAuthenticationChallenge *);

    id sender() const { return m_sender.get(); }
    NSURLAuthenticationChallenge *nsURLAuthenticationChallenge() const { return m_nsChallenge.get(); }

    WEBCORE_EXPORT void setAuthenticationClient(AuthenticationClient*); // Changes sender to one that invokes client methods.
    WEBCORE_EXPORT AuthenticationClient* authenticationClient() const;

private:
    friend class AuthenticationChallengeBase;
    static bool platformCompare(const AuthenticationChallenge& a, const AuthenticationChallenge& b);

    // Platform challenge may be null. If it's non-null, it's always up to date with other fields.
    RetainPtr<id> m_sender;
    RetainPtr<NSURLAuthenticationChallenge> m_nsChallenge;
};

}
