/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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

#if ENABLE(APPLE_PAY)

#include "ApplePaySessionPaymentRequest.h"
#include <wtf/Expected.h>
#include <wtf/Function.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {
class PaymentCoordinator;
}

namespace WebCore {

class ApplePaySetupFeature;
class Document;
class Payment;
class PaymentCoordinatorClient;
class PaymentContact;
class PaymentMerchantSession;
class PaymentMethod;
class PaymentSession;
class PaymentSessionError;
struct ApplePayCouponCodeUpdate;
struct ApplePayPaymentAuthorizationResult;
struct ApplePayPaymentMethodUpdate;
struct ApplePaySetupConfiguration;
struct ApplePayShippingContactUpdate;
struct ApplePayShippingMethod;
struct ApplePayShippingMethodUpdate;
struct ExceptionDetails;

class PaymentCoordinator final : public RefCountedAndCanMakeWeakPtr<PaymentCoordinator> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PaymentCoordinator, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static Ref<PaymentCoordinator> create(Ref<PaymentCoordinatorClient>&&);
    WEBCORE_EXPORT ~PaymentCoordinator();

    PaymentCoordinatorClient& client() { return m_client.get(); }

    bool supportsVersion(Document&, unsigned version) const;
    bool canMakePayments();
    void canMakePaymentsWithActiveCard(Document&, const String& merchantIdentifier, Function<void(bool)>&& completionHandler);
    void openPaymentSetup(Document&, const String& merchantIdentifier, Function<void(bool)>&& completionHandler);

    bool hasActiveSession() const { return m_activeSession; }

    bool beginPaymentSession(Document&, PaymentSession&, const ApplePaySessionPaymentRequest&);
    void completeMerchantValidation(const PaymentMerchantSession&);
    void completeShippingMethodSelection(std::optional<ApplePayShippingMethodUpdate>&&);
    void completeShippingContactSelection(std::optional<ApplePayShippingContactUpdate>&&);
    void completePaymentMethodSelection(std::optional<ApplePayPaymentMethodUpdate>&&);
#if ENABLE(APPLE_PAY_COUPON_CODE)
    void completeCouponCodeChange(std::optional<ApplePayCouponCodeUpdate>&&);
#endif
    void completePaymentSession(ApplePayPaymentAuthorizationResult&&);
    void abortPaymentSession();
    void cancelPaymentSession();

    WEBCORE_EXPORT void validateMerchant(URL&& validationURL);
    WEBCORE_EXPORT void didAuthorizePayment(const Payment&);
    WEBCORE_EXPORT void didSelectPaymentMethod(const PaymentMethod&);
    WEBCORE_EXPORT void didSelectShippingMethod(const ApplePayShippingMethod&);
    WEBCORE_EXPORT void didSelectShippingContact(const PaymentContact&);
#if ENABLE(APPLE_PAY_COUPON_CODE)
    WEBCORE_EXPORT void didChangeCouponCode(String&& couponCode);
#endif
    WEBCORE_EXPORT void didCancelPaymentSession(PaymentSessionError&&);

    std::optional<String> validatedPaymentNetwork(Document&, unsigned version, const String&) const;

    void getSetupFeatures(const ApplePaySetupConfiguration&, const URL&, CompletionHandler<void(Vector<Ref<ApplePaySetupFeature>>&&)>&&);
    void beginApplePaySetup(const ApplePaySetupConfiguration&, const URL&, Vector<Ref<ApplePaySetupFeature>>&&, CompletionHandler<void(bool)>&&);
    void endApplePaySetup();

protected:
    WEBCORE_EXPORT explicit PaymentCoordinator(Ref<PaymentCoordinatorClient>&&);

private:
    Ref<PaymentCoordinatorClient> m_client;
    RefPtr<PaymentSession> m_activeSession;
};

}

#endif
