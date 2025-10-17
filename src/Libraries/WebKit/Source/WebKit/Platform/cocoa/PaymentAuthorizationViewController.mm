/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#import "PaymentAuthorizationViewController.h"

#if USE(PASSKIT) && ENABLE(APPLE_PAY)

#import "WKPaymentAuthorizationDelegate.h"
#import <wtf/CompletionHandler.h>

#import <pal/cocoa/PassKitSoftLink.h>

@interface WKPaymentAuthorizationViewControllerDelegate : WKPaymentAuthorizationDelegate <PKPaymentAuthorizationViewControllerDelegate, PKPaymentAuthorizationViewControllerPrivateDelegate>

- (instancetype)initWithRequest:(PKPaymentRequest *)request presenter:(WebKit::PaymentAuthorizationPresenter&)presenter;

@end

@implementation WKPaymentAuthorizationViewControllerDelegate

- (instancetype)initWithRequest:(PKPaymentRequest *)request presenter:(WebKit::PaymentAuthorizationPresenter&)presenter
{
    if (!(self = [super _initWithRequest:request presenter:presenter]))
        return nil;

    return self;
}

- (void)_getPaymentServicesMerchantURL:(void(^)(NSURL *, NSError *))completion
{
    [PAL::getPKPaymentAuthorizationViewControllerClass() paymentServicesMerchantURLForAPIType:[_request APIType] completion:completion];
}

#pragma mark PKPaymentAuthorizationViewControllerDelegate

- (void)paymentAuthorizationViewControllerDidFinish:(PKPaymentAuthorizationViewController *)controller
{
    [self _didFinish];
}

- (void)paymentAuthorizationViewController:(PKPaymentAuthorizationViewController *)controller didAuthorizePayment:(PKPayment *)payment handler:(void (^)(PKPaymentAuthorizationResult *result))completion
{
    [self _didAuthorizePayment:payment completion:completion];
}

- (void)paymentAuthorizationViewController:(PKPaymentAuthorizationViewController *)controller didSelectShippingMethod:(PKShippingMethod *)shippingMethod handler:(void (^)(PKPaymentRequestShippingMethodUpdate *update))completion
{
    [self _didSelectShippingMethod:shippingMethod completion:completion];
}

- (void)paymentAuthorizationViewController:(PKPaymentAuthorizationViewController *)controller didSelectShippingContact:(PKContact *)contact handler:(void (^)(PKPaymentRequestShippingContactUpdate *update))completion
{
    [self _didSelectShippingContact:contact completion:completion];
}

- (void)paymentAuthorizationViewController:(PKPaymentAuthorizationViewController *)controller didSelectPaymentMethod:(PKPaymentMethod *)paymentMethod handler:(void (^)(PKPaymentRequestPaymentMethodUpdate *update))completion
{
    [self _didSelectPaymentMethod:paymentMethod completion:completion];
}

#if HAVE(PASSKIT_COUPON_CODE)

- (void)paymentAuthorizationViewController:(PKPaymentAuthorizationViewController *)controller didChangeCouponCode:(NSString *)couponCode handler:(void (^)(PKPaymentRequestCouponCodeUpdate *update))completion
{
    [self _didChangeCouponCode:couponCode completion:completion];
}

#endif // HAVE(PASSKIT_COUPON_CODE)

#pragma mark PKPaymentAuthorizationViewControllerDelegatePrivate

- (void)paymentAuthorizationViewController:(PKPaymentAuthorizationViewController *)controller willFinishWithError:(NSError *)error
{
    [self _willFinishWithError:error];
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (void)paymentAuthorizationViewController:(PKPaymentAuthorizationViewController *)controller didRequestMerchantSession:(void(^)(PKPaymentMerchantSession *, NSError *))completion
{
    [self _didRequestMerchantSession:completion];
}
ALLOW_DEPRECATED_IMPLEMENTATIONS_END

@end

namespace WebKit {

static RetainPtr<PKPaymentAuthorizationViewController> platformViewController(PKPaymentRequest *request, PKPaymentAuthorizationViewController *viewController)
{
#if PLATFORM(IOS_FAMILY)
    ASSERT(!viewController);
    return adoptNS([PAL::allocPKPaymentAuthorizationViewControllerInstance() initWithPaymentRequest:request]);
#else
    return viewController;
#endif
}

Ref<PaymentAuthorizationViewController> PaymentAuthorizationViewController::create(PaymentAuthorizationPresenter::Client& client, PKPaymentRequest *request, PKPaymentAuthorizationViewController *viewController)
{
    return adoptRef(*new PaymentAuthorizationViewController(client, request, viewController));
}

PaymentAuthorizationViewController::PaymentAuthorizationViewController(PaymentAuthorizationPresenter::Client& client, PKPaymentRequest *request, PKPaymentAuthorizationViewController *viewController)
    : PaymentAuthorizationPresenter(client)
    , m_viewController(platformViewController(request, viewController))
    , m_delegate(adoptNS([[WKPaymentAuthorizationViewControllerDelegate alloc] initWithRequest:request presenter:*this]))
{
    [m_viewController setDelegate:m_delegate.get()];
    [m_viewController setPrivateDelegate:m_delegate.get()];
}

WKPaymentAuthorizationDelegate *PaymentAuthorizationViewController::platformDelegate()
{
    return m_delegate.get();
}

void PaymentAuthorizationViewController::dismiss()
{
#if PLATFORM(IOS_FAMILY)
    [[m_viewController presentingViewController] dismissViewControllerAnimated:YES completion:nullptr];
#endif
    [m_viewController setDelegate:nil];
    [m_viewController setPrivateDelegate:nil];
    m_viewController = nil;
    [m_delegate invalidate];
    m_delegate = nil;
}

#if PLATFORM(IOS_FAMILY)

void PaymentAuthorizationViewController::present(UIViewController *presentingViewController, CompletionHandler<void(bool)>&& completionHandler)
{
    if (!presentingViewController || !m_viewController)
        return completionHandler(false);

    [presentingViewController presentViewController:m_viewController.get() animated:YES completion:nullptr];
    completionHandler(true);
}

#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
void PaymentAuthorizationViewController::presentInScene(const String&, const String&, CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT_NOT_REACHED();
    completionHandler(false);
}
#endif

#endif

} // namespace WebKit

#endif // USE(PASSKIT) && ENABLE(APPLE_PAY)
