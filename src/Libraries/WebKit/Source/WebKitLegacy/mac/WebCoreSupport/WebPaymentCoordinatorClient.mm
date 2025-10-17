/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#import "WebPaymentCoordinatorClient.h"

#if ENABLE(APPLE_PAY)

#import <wtf/CompletionHandler.h>
#import <wtf/MainThread.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/URL.h>

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPaymentCoordinatorClient);

Ref<WebPaymentCoordinatorClient> WebPaymentCoordinatorClient::create()
{
    return adoptRef(*new WebPaymentCoordinatorClient);
}

// FIXME: Why is this distinct from EmptyPaymentCoordinatorClient?
WebPaymentCoordinatorClient::WebPaymentCoordinatorClient() = default;

WebPaymentCoordinatorClient::~WebPaymentCoordinatorClient() = default;

std::optional<String> WebPaymentCoordinatorClient::validatedPaymentNetwork(const String&) const
{
    return std::nullopt;
}

bool WebPaymentCoordinatorClient::canMakePayments()
{
    return false;
}

void WebPaymentCoordinatorClient::canMakePaymentsWithActiveCard(const String&, const String&, CompletionHandler<void(bool)>&& completionHandler)
{
    callOnMainThread([completionHandler = WTFMove(completionHandler)]() mutable {
        completionHandler(false);
    });
}

void WebPaymentCoordinatorClient::openPaymentSetup(const String&, const String&, CompletionHandler<void(bool)>&& completionHandler)
{
    callOnMainThread([completionHandler = WTFMove(completionHandler)]() mutable {
        completionHandler(false);
    });
}

bool WebPaymentCoordinatorClient::showPaymentUI(const URL&, const Vector<URL>&, const WebCore::ApplePaySessionPaymentRequest&)
{
    return false;
}

void WebPaymentCoordinatorClient::completeMerchantValidation(const WebCore::PaymentMerchantSession&)
{
}

void WebPaymentCoordinatorClient::completeShippingMethodSelection(std::optional<WebCore::ApplePayShippingMethodUpdate>&&)
{
}

void WebPaymentCoordinatorClient::completeShippingContactSelection(std::optional<WebCore::ApplePayShippingContactUpdate>&&)
{
}

void WebPaymentCoordinatorClient::completePaymentMethodSelection(std::optional<WebCore::ApplePayPaymentMethodUpdate>&&)
{
}

#if ENABLE(APPLE_PAY_COUPON_CODE)

void WebPaymentCoordinatorClient::completeCouponCodeChange(std::optional<WebCore::ApplePayCouponCodeUpdate>&&)
{
}

#endif // ENABLE(APPLE_PAY_COUPON_CODE)

void WebPaymentCoordinatorClient::completePaymentSession(WebCore::ApplePayPaymentAuthorizationResult&&)
{
}

void WebPaymentCoordinatorClient::abortPaymentSession()
{
}

void WebPaymentCoordinatorClient::cancelPaymentSession()
{
}

#endif
