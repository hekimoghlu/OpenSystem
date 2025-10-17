/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#ifndef AuthenticationChallengeBase_h
#define AuthenticationChallengeBase_h

#include "Credential.h"
#include "ProtectionSpace.h"
#include "ResourceResponse.h"
#include "ResourceError.h"

namespace WebCore {

class AuthenticationChallenge;

class AuthenticationChallengeBase {
public:
    WEBCORE_EXPORT AuthenticationChallengeBase();
    WEBCORE_EXPORT AuthenticationChallengeBase(const ProtectionSpace& protectionSpace, const Credential& proposedCredential, unsigned previousFailureCount, const ResourceResponse& response, const ResourceError& error);

    WEBCORE_EXPORT unsigned previousFailureCount() const;
    WEBCORE_EXPORT const Credential& proposedCredential() const;
    WEBCORE_EXPORT const ProtectionSpace& protectionSpace() const;
    WEBCORE_EXPORT const ResourceResponse& failureResponse() const;
    WEBCORE_EXPORT const ResourceError& error() const;
    
    WEBCORE_EXPORT bool isNull() const;
    WEBCORE_EXPORT void nullify();
    
    WEBCORE_EXPORT static bool equalForWebKitLegacyChallengeComparison(const AuthenticationChallenge&, const AuthenticationChallenge&);

protected:
    // The AuthenticationChallenge subclass may "shadow" this method to compare platform specific fields
    static bool platformCompare(const AuthenticationChallengeBase&, const AuthenticationChallengeBase&) { return true; }

    bool m_isNull;
    ProtectionSpace m_protectionSpace;
    Credential m_proposedCredential;
    unsigned m_previousFailureCount;
    ResourceResponse m_failureResponse;
    ResourceError m_error;
};

}

#endif
