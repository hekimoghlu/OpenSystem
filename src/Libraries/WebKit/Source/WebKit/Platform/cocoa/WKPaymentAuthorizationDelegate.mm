/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
#import "WKPaymentAuthorizationDelegate.h"

#if USE(PASSKIT) && ENABLE(APPLE_PAY)

#import "PaymentAuthorizationPresenter.h"
#import <WebCore/ApplePayShippingMethod.h>
#import <WebCore/Payment.h>
#import <WebCore/PaymentMethod.h>
#import <WebCore/PaymentSessionError.h>
#import <wtf/RunLoop.h>
#import <wtf/URL.h>

#import <pal/cocoa/PassKitSoftLink.h>

@implementation WKPaymentAuthorizationDelegate {
    RetainPtr<NSArray<PKPaymentSummaryItem *>> _summaryItems;
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
    RetainPtr<PKShippingMethods> _availableShippingMethods;
#else
    RetainPtr<NSArray<PKShippingMethod *>> _shippingMethods;
#endif
    RetainPtr<NSError> _sessionError;
    WebKit::DidAuthorizePaymentCompletion _didAuthorizePaymentCompletion;
    WebKit::DidRequestMerchantSessionCompletion _didRequestMerchantSessionCompletion;
    WebKit::DidSelectPaymentMethodCompletion _didSelectPaymentMethodCompletion;
    WebKit::DidSelectShippingContactCompletion _didSelectShippingContactCompletion;
    WebKit::DidSelectShippingMethodCompletion _didSelectShippingMethodCompletion;
#if HAVE(PASSKIT_COUPON_CODE)
    WebKit::DidChangeCouponCodeCompletion _didChangeCouponCodeCompletion;
#endif
}

- (void)completeMerchantValidation:(PKPaymentMerchantSession *)session error:(NSError *)error
{
    std::exchange(_didRequestMerchantSessionCompletion, nil)(session, error);
}

- (void)completePaymentMethodSelection:(PKPaymentRequestPaymentMethodUpdate *)paymentMethodUpdate
{
    RetainPtr update = paymentMethodUpdate;
    if (update) {
        _summaryItems = adoptNS([[update paymentSummaryItems] copy]);
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
        _availableShippingMethods = adoptNS([[update availableShippingMethods] copy]);
#elif HAVE(PASSKIT_UPDATE_SHIPPING_METHODS_WHEN_CHANGING_SUMMARY_ITEMS)
        _shippingMethods = adoptNS([[update shippingMethods] copy]);
#endif
    } else {
        update = adoptNS([PAL::allocPKPaymentRequestPaymentMethodUpdateInstance() initWithPaymentSummaryItems:_summaryItems.get()]);
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
        [update setAvailableShippingMethods:_availableShippingMethods.get()];
#elif HAVE(PASSKIT_UPDATE_SHIPPING_METHODS_WHEN_CHANGING_SUMMARY_ITEMS)
        [update setShippingMethods:_shippingMethods.get()];
#endif
    }

    std::exchange(_didSelectPaymentMethodCompletion, nil)(update.get());
}

- (void)completePaymentSession:(PKPaymentAuthorizationStatus)status errors:(NSArray<NSError *> *)errors
{
    auto result = adoptNS([PAL::allocPKPaymentAuthorizationResultInstance() initWithStatus:status errors:errors]);
    std::exchange(_didAuthorizePaymentCompletion, nil)(result.get());
}

#if HAVE(PASSKIT_PAYMENT_ORDER_DETAILS)

- (void)completePaymentSession:(PKPaymentAuthorizationStatus)status errors:(NSArray<NSError *> *)errors orderDetails:(PKPaymentOrderDetails *)orderDetails
{
    auto result = adoptNS([PAL::allocPKPaymentAuthorizationResultInstance() initWithStatus:status errors:errors]);
    [result setOrderDetails:orderDetails];
    std::exchange(_didAuthorizePaymentCompletion, nil)(result.get());
}

#endif // HAVE(PASSKIT_PAYMENT_ORDER_DETAILS)

- (void)completeShippingContactSelection:(PKPaymentRequestShippingContactUpdate *)shippingContactUpdate
{
    RetainPtr update = shippingContactUpdate;
    if (update) {
        _summaryItems = adoptNS([[update paymentSummaryItems] copy]);
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
        _availableShippingMethods = adoptNS([[update availableShippingMethods] copy]);
#else
        _shippingMethods = adoptNS([[update shippingMethods] copy]);
#endif
    } else {
        update = adoptNS([PAL::allocPKPaymentRequestShippingContactUpdateInstance() initWithPaymentSummaryItems:_summaryItems.get()]);
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
        [update setAvailableShippingMethods:_availableShippingMethods.get()];
#else
        [update setShippingMethods:_shippingMethods.get()];
#endif
    }

    std::exchange(_didSelectShippingContactCompletion, nil)(update.get());
}

- (void)completeShippingMethodSelection:(PKPaymentRequestShippingMethodUpdate *)shippingMethodUpdate
{
    RetainPtr update = shippingMethodUpdate;
    if (update) {
        _summaryItems = adoptNS([[update paymentSummaryItems] copy]);
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
        _availableShippingMethods = adoptNS([[update availableShippingMethods] copy]);
#elif HAVE(PASSKIT_UPDATE_SHIPPING_METHODS_WHEN_CHANGING_SUMMARY_ITEMS)
        _shippingMethods = adoptNS([[update shippingMethods] copy]);
#endif
    } else {
        update = adoptNS([PAL::allocPKPaymentRequestShippingMethodUpdateInstance() initWithPaymentSummaryItems:_summaryItems.get()]);
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
        [update setAvailableShippingMethods:_availableShippingMethods.get()];
#elif HAVE(PASSKIT_UPDATE_SHIPPING_METHODS_WHEN_CHANGING_SUMMARY_ITEMS)
        [update setShippingMethods:_shippingMethods.get()];
#endif
    }

    std::exchange(_didSelectShippingMethodCompletion, nil)(update.get());
}

#if HAVE(PASSKIT_COUPON_CODE)

- (void)completeCouponCodeChange:(PKPaymentRequestCouponCodeUpdate *)couponCodeUpdate
{
    RetainPtr update = couponCodeUpdate;
    if (update) {
        _summaryItems = adoptNS([[update paymentSummaryItems] copy]);
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
        _availableShippingMethods = adoptNS([[update availableShippingMethods] copy]);
#else
        _shippingMethods = adoptNS([[update shippingMethods] copy]);
#endif
    } else {
        update = adoptNS([PAL::allocPKPaymentRequestCouponCodeUpdateInstance() initWithPaymentSummaryItems:_summaryItems.get()]);
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
        [update setAvailableShippingMethods:_availableShippingMethods.get()];
#else
        [update setShippingMethods:_shippingMethods.get()];
#endif
    }

    std::exchange(_didChangeCouponCodeCompletion, nil)(update.get());
}

#endif // HAVE(PASSKIT_COUPON_CODE)

- (void)invalidate
{
    if (_didAuthorizePaymentCompletion)
        [self completePaymentSession:PKPaymentAuthorizationStatusFailure errors:@[ ]];
}

@end

@implementation WKPaymentAuthorizationDelegate (Protected)

- (instancetype)_initWithRequest:(PKPaymentRequest *)request presenter:(WebKit::PaymentAuthorizationPresenter&)presenter
{
    if (!(self = [super init]))
        return nil;

    _presenter = presenter;
    _request = request;
#if HAVE(PASSKIT_DEFAULT_SHIPPING_METHOD)
    _availableShippingMethods = request.availableShippingMethods;
#else
    _shippingMethods = request.shippingMethods;
#endif
    _summaryItems = request.paymentSummaryItems;
    return self;
}

- (void)_didAuthorizePayment:(PKPayment *)payment completion:(WebKit::DidAuthorizePaymentCompletion::BlockType)completion
{
    ASSERT(!_didAuthorizePaymentCompletion);
    _didAuthorizePaymentCompletion = completion;

    auto presenter = _presenter.get();
    if (!presenter)
        return [self completePaymentSession:PKPaymentAuthorizationStatusFailure errors:@[ ]];

    RefPtr client = presenter->protectedClient();
    if (!client)
        return [self completePaymentSession:PKPaymentAuthorizationStatusFailure errors:@[ ]];

    client->presenterDidAuthorizePayment(*presenter, WebCore::Payment(payment));
}

- (void)_didFinish
{
    RefPtr presenter = _presenter.get();
    if (!presenter)
        return;

    RefPtr client = presenter->protectedClient();
    if (!client)
        return;

    client->presenterDidFinish(*presenter, { std::exchange(_sessionError, nil) });
}

- (void)_didRequestMerchantSession:(WebKit::DidRequestMerchantSessionCompletion::BlockType)completion
{
    ASSERT(!_didRequestMerchantSessionCompletion);
    _didRequestMerchantSessionCompletion = completion;

    [self _getPaymentServicesMerchantURL:^(NSURL *merchantURL, NSError *error) {
        if (error)
            LOG_ERROR("PKCanMakePaymentsWithMerchantIdentifierAndDomain error %@", error);

        RunLoop::main().dispatch([self, protectedSelf = retainPtr(self), merchantURL = retainPtr(merchantURL)] {
            ASSERT(_didRequestMerchantSessionCompletion);

            auto presenter = _presenter.get();
            if (!presenter) {
                _didRequestMerchantSessionCompletion(nil, nil);
                return;
            }

            RefPtr client = presenter->protectedClient();
            if (!client) {
                _didRequestMerchantSessionCompletion(nil, nil);
                return;
            }

            client->presenterWillValidateMerchant(*presenter, merchantURL.get());
        });
    }];
}

- (void)_didSelectPaymentMethod:(PKPaymentMethod *)paymentMethod completion:(WebKit::DidSelectPaymentMethodCompletion::BlockType)completion
{
    ASSERT(!_didSelectPaymentMethodCompletion);
    _didSelectPaymentMethodCompletion = completion;

    RefPtr presenter = _presenter.get();
    if (!presenter)
        return [self completePaymentMethodSelection:nil];

    RefPtr client = presenter->protectedClient();
    if (!client)
        return [self completePaymentMethodSelection:nil];

    client->presenterDidSelectPaymentMethod(*presenter, WebCore::PaymentMethod(paymentMethod));
}

- (void)_didSelectShippingContact:(PKContact *)contact completion:(WebKit::DidSelectShippingContactCompletion::BlockType)completion
{
    ASSERT(!_didSelectShippingContactCompletion);
    _didSelectShippingContactCompletion = completion;

    RefPtr presenter = _presenter.get();
    if (!presenter)
        return [self completeShippingContactSelection:nil];

    RefPtr client = presenter->protectedClient();
    if (!client)
        return [self completeShippingContactSelection:nil];

    client->presenterDidSelectShippingContact(*presenter, WebCore::PaymentContact(contact));
}

#if HAVE(PASSKIT_SHIPPING_METHOD_DATE_COMPONENTS_RANGE)

static WebCore::ApplePayDateComponents toDateComponents(NSDateComponents *dateComponents)
{
    ASSERT(dateComponents);

    WebCore::ApplePayDateComponents result;
    result.years = dateComponents.year;
    result.months = dateComponents.month;
    result.days = dateComponents.day;
    result.hours = dateComponents.hour;
    return result;
}

static WebCore::ApplePayDateComponentsRange toDateComponentsRange(PKDateComponentsRange *dateComponentsRange)
{
    ASSERT(dateComponentsRange);

    WebCore::ApplePayDateComponentsRange result;
    result.startDateComponents = toDateComponents(dateComponentsRange.startDateComponents);
    result.endDateComponents = toDateComponents(dateComponentsRange.endDateComponents);
    return result;
}

#endif // HAVE(PASSKIT_SHIPPING_METHOD_DATE_COMPONENTS_RANGE)

static WebCore::ApplePayShippingMethod toShippingMethod(PKShippingMethod *shippingMethod, bool selected)
{
    ASSERT(shippingMethod);

    WebCore::ApplePayShippingMethod result;
    result.amount = shippingMethod.amount.stringValue;
    result.detail = shippingMethod.detail;
    result.identifier = shippingMethod.identifier;
    result.label = shippingMethod.label;
#if HAVE(PASSKIT_SHIPPING_METHOD_DATE_COMPONENTS_RANGE)
    if (shippingMethod.dateComponentsRange)
        result.dateComponentsRange = toDateComponentsRange(shippingMethod.dateComponentsRange);
#endif
#if ENABLE(APPLE_PAY_SELECTED_SHIPPING_METHOD)
    result.selected = selected;
#else
    UNUSED_PARAM(selected);
#endif
    return result;
}

- (void)_didSelectShippingMethod:(PKShippingMethod *)shippingMethod completion:(WebKit::DidSelectShippingMethodCompletion::BlockType)completion
{
    ASSERT(!_didSelectShippingMethodCompletion);
    _didSelectShippingMethodCompletion = completion;

    RefPtr presenter = _presenter.get();
    if (!presenter)
        return [self completeShippingMethodSelection:nil];

    RefPtr client = presenter->protectedClient();
    if (!client)
        return [self completeShippingMethodSelection:nil];

    client->presenterDidSelectShippingMethod(*presenter, toShippingMethod(shippingMethod, true));
}

#if HAVE(PASSKIT_COUPON_CODE)

- (void)_didChangeCouponCode:(NSString *)couponCode completion:(void (^)(PKPaymentRequestCouponCodeUpdate *update))completion
{
    ASSERT(!_didChangeCouponCodeCompletion);
    _didChangeCouponCodeCompletion = completion;

    RefPtr presenter = _presenter.get();
    if (!presenter)
        return [self completeCouponCodeChange:nil];

    RefPtr client = presenter->protectedClient();
    if (!client)
        return [self completeCouponCodeChange:nil];

    client->presenterDidChangeCouponCode(*presenter, couponCode);
}

#endif // HAVE(PASSKIT_COUPON_CODE)

- (void) NO_RETURN_DUE_TO_ASSERT _getPaymentServicesMerchantURL:(void(^)(NSURL *, NSError *))completion
{
    ASSERT_NOT_REACHED();
    completion(nil, nil);
}

- (void)_willFinishWithError:(NSError *)error
{
    if (![error.domain isEqualToString:PKPassKitErrorDomain])
        return;

    _sessionError = error;
}

@end

#endif // USE(PASSKIT) && ENABLE(APPLE_PAY)
