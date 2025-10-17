/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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

namespace WebCore {

AuthenticationChallengeBase::AuthenticationChallengeBase()
    : m_isNull(true)
    , m_previousFailureCount(0)
{
}

AuthenticationChallengeBase::AuthenticationChallengeBase(const ProtectionSpace& protectionSpace,
                                                         const Credential& proposedCredential,
                                                         unsigned previousFailureCount,
                                                         const ResourceResponse& response,
                                                         const ResourceError& error)
    : m_isNull(false)
    , m_protectionSpace(protectionSpace)
    , m_proposedCredential(proposedCredential)
    , m_previousFailureCount(previousFailureCount)
    , m_failureResponse(response)
    , m_error(error)
{
}

unsigned AuthenticationChallengeBase::previousFailureCount() const 
{ 
    return m_previousFailureCount; 
}

const Credential& AuthenticationChallengeBase::proposedCredential() const 
{ 
    return m_proposedCredential; 
}

const ProtectionSpace& AuthenticationChallengeBase::protectionSpace() const 
{ 
    return m_protectionSpace; 
}

const ResourceResponse& AuthenticationChallengeBase::failureResponse() const 
{ 
    return m_failureResponse; 
}

const ResourceError& AuthenticationChallengeBase::error() const 
{ 
    return m_error; 
}

bool AuthenticationChallengeBase::isNull() const
{
    return m_isNull;
}

void AuthenticationChallengeBase::nullify()
{
    m_isNull = true;
}

bool AuthenticationChallengeBase::equalForWebKitLegacyChallengeComparison(const AuthenticationChallenge& a, const AuthenticationChallenge& b)
{
    if (a.isNull() && b.isNull())
        return true;

    if (a.isNull() || b.isNull())
        return false;
        
    if (a.protectionSpace() != b.protectionSpace())
        return false;
        
    if (a.proposedCredential() != b.proposedCredential())
        return false;
        
    if (a.previousFailureCount() != b.previousFailureCount())
        return false;
        
    if (!ResourceResponseBase::equalForWebKitLegacyChallengeComparison(a.failureResponse(), b.failureResponse()))
        return false;

    if (a.error() != b.error())
        return false;
        
    return AuthenticationChallenge::platformCompare(a, b);
}

}
