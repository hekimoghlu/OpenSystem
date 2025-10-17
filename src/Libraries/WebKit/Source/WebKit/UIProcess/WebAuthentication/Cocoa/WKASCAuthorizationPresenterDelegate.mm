/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#import "WKASCAuthorizationPresenterDelegate.h"

#if ENABLE(WEB_AUTHN) && HAVE(ASC_AUTH_UI)

#import "AuthenticatorPresenterCoordinator.h"

#import <wtf/BlockPtr.h>
#import <wtf/RunLoop.h>
#import "AuthenticationServicesCoreSoftLink.h"

NS_ASSUME_NONNULL_BEGIN

@implementation WKASCAuthorizationPresenterDelegate {
    WeakPtr<WebKit::AuthenticatorPresenterCoordinator> _coordinator;
}

- (instancetype)initWithCoordinator:(WebKit::AuthenticatorPresenterCoordinator&)coordinator
{
    if ((self = [super init]))
        _coordinator = coordinator;
    return self;
}

- (void)authorizationPresenter:(ASCAuthorizationPresenter *)presenter credentialRequestedForLoginChoice:(id <ASCLoginChoiceProtocol>)loginChoice authenticatedContext:(nullable LAContext *)context completionHandler:(void (^)(id <ASCCredentialProtocol> _Nullable credential, NSError * _Nullable error))completionHandler
{
    auto requestHandler = [completionHandler = makeBlockPtr(completionHandler)] (ASCAppleIDCredential *credential, NSError *error) {
        completionHandler(credential, error);
    };
    [self dispatchCoordinatorCallback:[requestHandler = WTFMove(requestHandler)] (WebKit::AuthenticatorPresenterCoordinator& coordinator) mutable {
        coordinator.setCredentialRequestHandler(WTFMove(requestHandler));
    }];

    if ([loginChoice isKindOfClass:WebKit::getASCPlatformPublicKeyCredentialLoginChoiceClass()]) {
        auto *platformLoginChoice = (ASCPlatformPublicKeyCredentialLoginChoice *)loginChoice;

        if ([platformLoginChoice isRegistrationRequest]) {
            [self dispatchCoordinatorCallback:[context = retainPtr(context)] (WebKit::AuthenticatorPresenterCoordinator& coordinator) mutable {
                coordinator.setLAContext(context.get());
            }];

            return;
        }

        String loginChoiceName = platformLoginChoice.name;
        [self dispatchCoordinatorCallback:[loginChoiceName = WTFMove(loginChoiceName), context = retainPtr(context)] (WebKit::AuthenticatorPresenterCoordinator& coordinator) mutable {
            coordinator.didSelectAssertionResponse(loginChoiceName, context.get());
        }];

        return;
    }

    if ([loginChoice isKindOfClass:WebKit::getASCSecurityKeyPublicKeyCredentialLoginChoiceClass()]) {
        auto *securityKeyLoginChoice = (ASCSecurityKeyPublicKeyCredentialLoginChoice *)loginChoice;

        if ([securityKeyLoginChoice credentialKind] == ASCSecurityKeyPublicKeyCredentialKindAssertion) {
            String loginChoiceName = securityKeyLoginChoice.name;
            [self dispatchCoordinatorCallback:[loginChoiceName = WTFMove(loginChoiceName)] (WebKit::AuthenticatorPresenterCoordinator& coordinator) mutable {
                coordinator.didSelectAssertionResponse(loginChoiceName, nil);
            }];

            return;
        }
    }
}

- (void)authorizationPresenter:(ASCAuthorizationPresenter *)presenter validateUserEnteredPIN:(NSString *)pin completionHandler:(void (^)(id <ASCCredentialProtocol> credential, NSError *error))completionHandler
{
    auto requestHandler = [completionHandler = makeBlockPtr(completionHandler)] (ASCAppleIDCredential *credential, NSError *error) {
        completionHandler(credential, error);
    };
    [self dispatchCoordinatorCallback:[requestHandler = WTFMove(requestHandler)] (WebKit::AuthenticatorPresenterCoordinator& coordinator) mutable {
        coordinator.setCredentialRequestHandler(WTFMove(requestHandler));
    }];

    String pinString = pin;
    [self dispatchCoordinatorCallback:[pinString = WTFMove(pinString)] (WebKit::AuthenticatorPresenterCoordinator& coordinator) mutable {
        coordinator.setPin(pinString);
    }];
}

- (void)dispatchCoordinatorCallback:(Function<void(WebKit::AuthenticatorPresenterCoordinator&)>&&)callback
{
    ASSERT(!RunLoop::isMain());
    RunLoop::main().dispatch([coordinator = _coordinator, callback = WTFMove(callback)] {
        if (!coordinator)
            return;
        callback(*coordinator);
    });
}

@end

NS_ASSUME_NONNULL_END

#endif // ENABLE(WEB_AUTHN) && HAVE(ASC_AUTH_UI)
