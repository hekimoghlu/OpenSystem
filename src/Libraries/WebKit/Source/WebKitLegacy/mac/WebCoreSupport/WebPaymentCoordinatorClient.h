/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#pragma once

#import <WebCore/PaymentCoordinatorClient.h>

#if ENABLE(APPLE_PAY)

#import <wtf/TZoneMalloc.h>

class WebPaymentCoordinatorClient final : public WebCore::PaymentCoordinatorClient, public RefCounted<WebPaymentCoordinatorClient> {
    WTF_MAKE_TZONE_ALLOCATED(WebPaymentCoordinatorClient);
public:
    static Ref<WebPaymentCoordinatorClient> create();
    ~WebPaymentCoordinatorClient();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    WebPaymentCoordinatorClient();

    std::optional<String> validatedPaymentNetwork(const String&) const override;
    bool canMakePayments() override;
    void canMakePaymentsWithActiveCard(const String&, const String&, CompletionHandler<void(bool)>&&) override;
    void openPaymentSetup(const String& merchantIdentifier, const String& domainName, CompletionHandler<void(bool)>&&) override;
    bool showPaymentUI(const URL&, const Vector<URL>& linkIconURLs, const WebCore::ApplePaySessionPaymentRequest&) override;
    void completeMerchantValidation(const WebCore::PaymentMerchantSession&) override;
    void completeShippingMethodSelection(std::optional<WebCore::ApplePayShippingMethodUpdate>&&) override;
    void completeShippingContactSelection(std::optional<WebCore::ApplePayShippingContactUpdate>&&) override;
    void completePaymentMethodSelection(std::optional<WebCore::ApplePayPaymentMethodUpdate>&&) override;
#if ENABLE(APPLE_PAY_COUPON_CODE)
    void completeCouponCodeChange(std::optional<WebCore::ApplePayCouponCodeUpdate>&&) override;
#endif
    void completePaymentSession(WebCore::ApplePayPaymentAuthorizationResult&&) override;
    void abortPaymentSession() override;
    void cancelPaymentSession() override;
};

#endif
