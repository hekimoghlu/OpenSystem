/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
#import "config.h"
#import "AuthenticationMac.h"

#import "AuthenticationChallenge.h"
#import "AuthenticationClient.h"
#import "Credential.h"
#import <Foundation/NSURLAuthenticationChallenge.h>
#import <Foundation/NSURLProtectionSpace.h>

using namespace WebCore;

@interface WebCoreAuthenticationClientAsChallengeSender : NSObject <NSURLAuthenticationChallengeSender>
{
    AuthenticationClient* m_client;
}
- (id)initWithAuthenticationClient:(AuthenticationClient*)client;
- (AuthenticationClient*)client;
- (void)detachClient;
@end

@implementation WebCoreAuthenticationClientAsChallengeSender

- (id)initWithAuthenticationClient:(AuthenticationClient*)client
{
    self = [self init];
    if (!self)
        return nil;
    m_client = client;
    return self;
}

- (AuthenticationClient*)client
{
    return m_client;
}

- (void)detachClient
{
    m_client = 0;
}

- (void)performDefaultHandlingForAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    if (m_client)
        m_client->receivedRequestToPerformDefaultHandling(core(challenge));
}

- (void)rejectProtectionSpaceAndContinueWithChallenge:(NSURLAuthenticationChallenge *)challenge
{
    if (m_client)
        m_client->receivedChallengeRejection(core(challenge));
}

- (void)useCredential:(NSURLCredential *)credential forAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    if (m_client)
        m_client->receivedCredential(core(challenge), Credential(credential));
}

- (void)continueWithoutCredentialForAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    if (m_client)
        m_client->receivedRequestToContinueWithoutCredential(core(challenge));
}

- (void)cancelAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    if (m_client)
        m_client->receivedCancellation(core(challenge));
}

@end

namespace WebCore {

AuthenticationChallenge::AuthenticationChallenge(const ProtectionSpace& protectionSpace,
                                                 const Credential& proposedCredential,
                                                 unsigned previousFailureCount,
                                                 const ResourceResponse& response,
                                                 const ResourceError& error)
    : AuthenticationChallengeBase(protectionSpace,
                                  proposedCredential,
                                  previousFailureCount,
                                  response,
                                  error)
{
}

AuthenticationChallenge::AuthenticationChallenge(NSURLAuthenticationChallenge *challenge)
    : AuthenticationChallengeBase(ProtectionSpace([challenge protectionSpace]),
                                  Credential([challenge proposedCredential]),
                                  [challenge previousFailureCount],
                                  [challenge failureResponse],
                                  [challenge error])
    , m_sender([challenge sender])
    , m_nsChallenge(challenge)
{
}

void AuthenticationChallenge::setAuthenticationClient(AuthenticationClient* client)
{
    if (client) {
        m_sender = adoptNS([[WebCoreAuthenticationClientAsChallengeSender alloc] initWithAuthenticationClient:client]);
        if (m_nsChallenge)
            m_nsChallenge = adoptNS([[NSURLAuthenticationChallenge alloc] initWithAuthenticationChallenge:m_nsChallenge.get() sender:m_sender.get()]);
    } else {
        if ([m_sender isMemberOfClass:[WebCoreAuthenticationClientAsChallengeSender class]])
            [(WebCoreAuthenticationClientAsChallengeSender *)m_sender.get() detachClient];
    }
}

AuthenticationClient* AuthenticationChallenge::authenticationClient() const
{
    if ([m_sender isMemberOfClass:[WebCoreAuthenticationClientAsChallengeSender class]])
        return [static_cast<WebCoreAuthenticationClientAsChallengeSender*>(m_sender.get()) client];
    
    return nullptr;
}

bool AuthenticationChallenge::platformCompare(const AuthenticationChallenge& a, const AuthenticationChallenge& b)
{
    if (a.sender() != b.sender())
        return false;
        
    if (a.nsURLAuthenticationChallenge() != b.nsURLAuthenticationChallenge())
        return false;

    return true;
}

NSURLAuthenticationChallenge *mac(const AuthenticationChallenge& coreChallenge)
{
    if (coreChallenge.nsURLAuthenticationChallenge())
        return coreChallenge.nsURLAuthenticationChallenge();
        
    return adoptNS([[NSURLAuthenticationChallenge alloc] initWithProtectionSpace:coreChallenge.protectionSpace().nsSpace() proposedCredential:coreChallenge.proposedCredential().nsCredential() previousFailureCount:coreChallenge.previousFailureCount() failureResponse:coreChallenge.failureResponse().nsURLResponse() error:coreChallenge.error() sender:coreChallenge.sender()]).autorelease();
}

AuthenticationChallenge core(NSURLAuthenticationChallenge *macChallenge)
{
    return AuthenticationChallenge(macChallenge);
}

} // namespace WebCore
