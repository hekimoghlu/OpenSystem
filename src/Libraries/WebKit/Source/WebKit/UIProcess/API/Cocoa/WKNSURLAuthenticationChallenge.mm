/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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
#import "WKNSURLAuthenticationChallenge.h"

#import "AuthenticationChallengeDisposition.h"
#import "AuthenticationDecisionListener.h"
#import "WebCredential.h"
#import <WebCore/AuthenticationMac.h>

@interface WKNSURLAuthenticationChallengeSender : NSObject <NSURLAuthenticationChallengeSender>
@end

@implementation WKNSURLAuthenticationChallenge

- (NSObject *)_web_createTarget
{
    WebKit::AuthenticationChallengeProxy& challenge = *reinterpret_cast<WebKit::AuthenticationChallengeProxy*>(&self._apiObject);

    static dispatch_once_t token;
    static NeverDestroyed<RetainPtr<WKNSURLAuthenticationChallengeSender>> sender;
    dispatch_once(&token, ^{
        sender.get() = adoptNS([[WKNSURLAuthenticationChallengeSender alloc] init]);
    });

    return [[NSURLAuthenticationChallenge alloc] initWithAuthenticationChallenge:mac(challenge.core()) sender:sender.get().get()];
}

- (WebKit::AuthenticationChallengeProxy&)_web_authenticationChallengeProxy
{
    return *reinterpret_cast<WebKit::AuthenticationChallengeProxy*>(&self._apiObject);
}

@end

@implementation WKNSURLAuthenticationChallengeSender

static void checkChallenge(NSURLAuthenticationChallenge *challenge)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if ([challenge class] != [WKNSURLAuthenticationChallenge class])
        [NSException raise:NSInvalidArgumentException format:@"The challenge was not sent by the receiver."];
ALLOW_DEPRECATED_DECLARATIONS_END
}

- (void)cancelAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    checkChallenge(challenge);
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WebKit::AuthenticationChallengeProxy& webChallenge = ((WKNSURLAuthenticationChallenge *)challenge)._web_authenticationChallengeProxy;
ALLOW_DEPRECATED_DECLARATIONS_END
    webChallenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::Cancel);
}

- (void)continueWithoutCredentialForAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    checkChallenge(challenge);
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WebKit::AuthenticationChallengeProxy& webChallenge = ((WKNSURLAuthenticationChallenge *)challenge)._web_authenticationChallengeProxy;
ALLOW_DEPRECATED_DECLARATIONS_END
    webChallenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::UseCredential);
}

- (void)useCredential:(NSURLCredential *)credential forAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    checkChallenge(challenge);
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WebKit::AuthenticationChallengeProxy& webChallenge = ((WKNSURLAuthenticationChallenge *)challenge)._web_authenticationChallengeProxy;
ALLOW_DEPRECATED_DECLARATIONS_END
    webChallenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::UseCredential, WebCore::Credential(credential));
}

- (void)performDefaultHandlingForAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    checkChallenge(challenge);
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WebKit::AuthenticationChallengeProxy& webChallenge = ((WKNSURLAuthenticationChallenge *)challenge)._web_authenticationChallengeProxy;
ALLOW_DEPRECATED_DECLARATIONS_END
    webChallenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::PerformDefaultHandling);
}

- (void)rejectProtectionSpaceAndContinueWithChallenge:(NSURLAuthenticationChallenge *)challenge
{
    checkChallenge(challenge);
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    WebKit::AuthenticationChallengeProxy& webChallenge = ((WKNSURLAuthenticationChallenge *)challenge)._web_authenticationChallengeProxy;
ALLOW_DEPRECATED_DECLARATIONS_END
    webChallenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue);
}

@end
