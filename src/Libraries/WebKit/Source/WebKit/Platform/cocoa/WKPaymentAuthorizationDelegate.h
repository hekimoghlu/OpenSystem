/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#if USE(PASSKIT)

#import <pal/spi/cocoa/PassKitSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakPtr.h>

OBJC_CLASS NSArray;
OBJC_CLASS NSError;
OBJC_CLASS UIViewController;

namespace WebKit {

class PaymentAuthorizationPresenter;

using DidRequestMerchantSessionCompletion = BlockPtr<void(PKPaymentMerchantSession *, NSError *)>;

using DidAuthorizePaymentCompletion = BlockPtr<void(PKPaymentAuthorizationResult *)>;
using DidSelectPaymentMethodCompletion = BlockPtr<void(PKPaymentRequestPaymentMethodUpdate *)>;
using DidSelectShippingContactCompletion = BlockPtr<void(PKPaymentRequestShippingContactUpdate *)>;
using DidSelectShippingMethodCompletion = BlockPtr<void(PKPaymentRequestShippingMethodUpdate *)>;
#if HAVE(PASSKIT_COUPON_CODE)
using DidChangeCouponCodeCompletion = BlockPtr<void(PKPaymentRequestCouponCodeUpdate *)>;
#endif

}

@interface WKPaymentAuthorizationDelegate : NSObject {
    RetainPtr<PKPaymentRequest> _request;
    WeakPtr<WebKit::PaymentAuthorizationPresenter> _presenter;
}

- (instancetype)init NS_UNAVAILABLE;

- (void)completeMerchantValidation:(PKPaymentMerchantSession *)session error:(NSError *)error;
- (void)completePaymentMethodSelection:(PKPaymentRequestPaymentMethodUpdate *)paymentMethodUpdate;
- (void)completePaymentSession:(PKPaymentAuthorizationStatus)status errors:(NSArray<NSError *> *)errors;
#if HAVE(PASSKIT_PAYMENT_ORDER_DETAILS)
- (void)completePaymentSession:(PKPaymentAuthorizationStatus)status errors:(NSArray<NSError *> *)errors orderDetails:(PKPaymentOrderDetails *)orderDetails;
#endif
- (void)completeShippingContactSelection:(PKPaymentRequestShippingContactUpdate *)shippingContactUpdate;
- (void)completeShippingMethodSelection:(PKPaymentRequestShippingMethodUpdate *)shippingMethodUpdate;
#if HAVE(PASSKIT_COUPON_CODE)
- (void)completeCouponCodeChange:(PKPaymentRequestCouponCodeUpdate *)couponCodeUpdate;
#endif
- (void)invalidate;

@end

// Helper methods called by WKPaymentAuthorizationSession's subclasses
@interface WKPaymentAuthorizationDelegate (Protected)

- (instancetype)_initWithRequest:(PKPaymentRequest *)request presenter:(WebKit::PaymentAuthorizationPresenter&)presenter;

- (void)_didAuthorizePayment:(PKPayment *)payment completion:(WebKit::DidAuthorizePaymentCompletion::BlockType)completion;
- (void)_didFinish;
- (void)_didRequestMerchantSession:(WebKit::DidRequestMerchantSessionCompletion::BlockType)completion;
- (void)_didSelectPaymentMethod:(PKPaymentMethod *)paymentMethod completion:(WebKit::DidSelectPaymentMethodCompletion::BlockType)completion;
- (void)_didSelectShippingContact:(PKContact *)contact completion:(WebKit::DidSelectShippingContactCompletion::BlockType)completion;
- (void)_didSelectShippingMethod:(PKShippingMethod *)shippingMethod completion:(WebKit::DidSelectShippingMethodCompletion::BlockType)completion;
#if HAVE(PASSKIT_COUPON_CODE)
- (void)_didChangeCouponCode:(NSString *)couponCode completion:(WebKit::DidChangeCouponCodeCompletion::BlockType)completion;
#endif
- (void)_getPaymentServicesMerchantURL:(void(^)(NSURL *, NSError *))completion;
- (void)_willFinishWithError:(NSError *)error;

@end

#endif // USE(PASSKIT)
