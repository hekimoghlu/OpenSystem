/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#import "PaymentAuthorizationController.h"

#if USE(PASSKIT) && PLATFORM(IOS_FAMILY)

#import "WKPaymentAuthorizationDelegate.h"
#import <wtf/CompletionHandler.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

#import <pal/cocoa/PassKitSoftLink.h>

@interface WKPaymentAuthorizationControllerDelegate : WKPaymentAuthorizationDelegate <PKPaymentAuthorizationControllerDelegate, PKPaymentAuthorizationControllerPrivateDelegate>

- (instancetype)initWithRequest:(PKPaymentRequest *)request presenter:(WebKit::PaymentAuthorizationPresenter&)presenter;

@end

@implementation WKPaymentAuthorizationControllerDelegate {
    __weak UIWindow *_presentingWindow;
}

- (instancetype)initWithRequest:(PKPaymentRequest *)request presenter:(WebKit::PaymentAuthorizationPresenter&)presenter
{
    if (!(self = [super _initWithRequest:request presenter:presenter]))
        return nil;

    RefPtr client = presenter.protectedClient();
    if (!client)
        return nil;

    _presentingWindow = client->presentingWindowForPaymentAuthorization(presenter);
    return self;
}

- (void)_getPaymentServicesMerchantURL:(void(^)(NSURL *, NSError *))completion
{
    // FIXME: This -respondsToSelector: check can be removed once rdar://problem/48771320 is in an iOS SDK.
    if ([PAL::getPKPaymentAuthorizationControllerClass() respondsToSelector:@selector(paymentServicesMerchantURLForAPIType:completion:)])
        [PAL::getPKPaymentAuthorizationControllerClass() paymentServicesMerchantURLForAPIType:[_request APIType] completion:completion];
    else
        [PAL::getPKPaymentAuthorizationViewControllerClass() paymentServicesMerchantURLForAPIType:[_request APIType] completion:completion];
}

#pragma mark PKPaymentAuthorizationControllerDelegate

- (void)paymentAuthorizationControllerDidFinish:(PKPaymentAuthorizationController *)controller
{
    [self _didFinish];
}

- (void)paymentAuthorizationController:(PKPaymentAuthorizationController *)controller didAuthorizePayment:(PKPayment *)payment handler:(void(^)(PKPaymentAuthorizationResult *result))completion
{
    [self _didAuthorizePayment:payment completion:completion];
}

- (void)paymentAuthorizationController:(PKPaymentAuthorizationController *)controller didSelectShippingMethod:(PKShippingMethod *)shippingMethod handler:(void(^)(PKPaymentRequestShippingMethodUpdate *requestUpdate))completion
{
    [self _didSelectShippingMethod:shippingMethod completion:completion];
}

- (void)paymentAuthorizationController:(PKPaymentAuthorizationController *)controller didSelectShippingContact:(PKContact *)contact handler:(void(^)(PKPaymentRequestShippingContactUpdate *requestUpdate))completion
{
    [self _didSelectShippingContact:contact completion:completion];
}

- (void)paymentAuthorizationController:(PKPaymentAuthorizationController *)controller didSelectPaymentMethod:(PKPaymentMethod *)paymentMethod handler:(void(^)(PKPaymentRequestPaymentMethodUpdate *requestUpdate))completion
{
    [self _didSelectPaymentMethod:paymentMethod completion:completion];
}

- (UIWindow *)presentationWindowForPaymentAuthorizationController:(PKPaymentAuthorizationController *)controller
{
    return _presentingWindow;
}

#if HAVE(PASSKIT_COUPON_CODE)

- (void)paymentAuthorizationController:(PKPaymentAuthorizationController *)controller didChangeCouponCode:(NSString *)couponCode handler:(void (^)(PKPaymentRequestCouponCodeUpdate *update))completion
{
    [self _didChangeCouponCode:couponCode completion:completion];
}

#endif // HAVE(PASSKIT_COUPON_CODE)

#pragma mark PKPaymentAuthorizationControllerPrivateDelegate

- (void)paymentAuthorizationController:(PKPaymentAuthorizationController *)controller willFinishWithError:(NSError *)error
{
    [self _willFinishWithError:error];
}

- (void)paymentAuthorizationController:(PKPaymentAuthorizationController *)controller didRequestMerchantSession:(void(^)(PKPaymentMerchantSession *, NSError *))sessionBlock
{
    [self _didRequestMerchantSession:sessionBlock];
}

#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
- (NSString *)presentationSceneIdentifierForPaymentAuthorizationController:(PKPaymentAuthorizationController *)controller
{
    if (!_presenter)
        return nil;
    return nsStringNilIfEmpty(_presenter->sceneIdentifier());
}

- (NSString *)presentationSceneBundleIdentifierForPaymentAuthorizationController:(PKPaymentAuthorizationController *)controller
{
    if (!_presenter)
        return applicationBundleIdentifier();
    return nsStringNilIfEmpty(_presenter->bundleIdentifier());
}
#endif

@end

namespace WebKit {

Ref<PaymentAuthorizationController> PaymentAuthorizationController::create(PaymentAuthorizationPresenter::Client& client, PKPaymentRequest *request)
{
    return adoptRef(*new PaymentAuthorizationController(client, request));
}

PaymentAuthorizationController::PaymentAuthorizationController(PaymentAuthorizationPresenter::Client& client, PKPaymentRequest *request)
    : PaymentAuthorizationPresenter(client)
    , m_controller(adoptNS([PAL::allocPKPaymentAuthorizationControllerInstance() initWithPaymentRequest:request]))
    , m_delegate(adoptNS([[WKPaymentAuthorizationControllerDelegate alloc] initWithRequest:request presenter:*this]))
{
    [m_controller setDelegate:m_delegate.get()];
    [m_controller setPrivateDelegate:m_delegate.get()];
}

WKPaymentAuthorizationDelegate *PaymentAuthorizationController::platformDelegate()
{
    return m_delegate.get();
}

void PaymentAuthorizationController::dismiss()
{
    [m_controller dismissWithCompletion:nil];
    [m_controller setDelegate:nil];
    [m_controller setPrivateDelegate:nil];
    m_controller = nil;
    [m_delegate invalidate];
    m_delegate = nil;
#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
    m_sceneIdentifier = nullString();
#endif
}

void PaymentAuthorizationController::present(UIViewController *, CompletionHandler<void(bool)>&& completionHandler)
{
    if (!m_controller)
        return completionHandler(false);

    [m_controller presentWithCompletion:makeBlockPtr([completionHandler = WTFMove(completionHandler)](BOOL success) mutable {
        completionHandler(success);
    }).get()];
}

#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
void PaymentAuthorizationController::presentInScene(const String& sceneIdentifier, const String& bundleIdentifier, CompletionHandler<void(bool)>&& completionHandler)
{
    m_sceneIdentifier = sceneIdentifier;
    m_bundleIdentifier = bundleIdentifier;
    present(nil, WTFMove(completionHandler));
}
#endif

} // namespace WebKit

#endif // USE(PASSKIT) && PLATFORM(IOS_FAMILY)
