/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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

#include "ApplePayInstallmentConfigurationWebCore.h"
#include "ApplePayLaterAvailability.h"
#include "ApplePayLineItem.h"
#include "ApplePaySetupConfiguration.h"
#include "ApplePayShippingContactEditingMode.h"
#include "ApplePayShippingMethod.h"
#include "MockPaymentAddress.h"
#include "MockPaymentContactFields.h"
#include "MockPaymentError.h"
#include "PaymentCoordinatorClient.h"
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class ApplePaySessionPaymentRequest;
class Page;
struct ApplePayDetailsUpdateBase;
struct ApplePayPaymentMethod;

class MockPaymentCoordinator final : public PaymentCoordinatorClient, public RefCounted<MockPaymentCoordinator> {
    WTF_MAKE_TZONE_ALLOCATED(MockPaymentCoordinator);
public:
    static Ref<MockPaymentCoordinator> create(Page&);
    ~MockPaymentCoordinator();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void setCanMakePayments(bool canMakePayments) { m_canMakePayments = canMakePayments; }
    void setCanMakePaymentsWithActiveCard(bool canMakePaymentsWithActiveCard) { m_canMakePaymentsWithActiveCard = canMakePaymentsWithActiveCard; }
    void setShippingAddress(MockPaymentAddress&& shippingAddress) { m_shippingAddress = WTFMove(shippingAddress); }
    void changeShippingOption(String&& shippingOption);
    void changePaymentMethod(ApplePayPaymentMethod&&);
#if ENABLE(APPLE_PAY_COUPON_CODE)
    void changeCouponCode(String&& couponCode);
#endif
    void acceptPayment();
    void cancelPayment();

    void addSetupFeature(ApplePaySetupFeatureState, ApplePaySetupFeatureType, bool supportsInstallments);
    const ApplePaySetupConfiguration& setupConfiguration() const { return m_setupConfiguration; }

    const ApplePayLineItem& total() const { return m_total; }
    const Vector<ApplePayLineItem>& lineItems() const { return m_lineItems; }
    const Vector<MockPaymentError>& errors() const { return m_errors; }
    const Vector<ApplePayShippingMethod>& shippingMethods() const { return m_shippingMethods; }
    const Vector<String>& supportedCountries() const { return m_supportedCountries; }
    const MockPaymentContactFields& requiredBillingContactFields() const { return m_requiredBillingContactFields; }
    const MockPaymentContactFields& requiredShippingContactFields() const { return m_requiredShippingContactFields; }

#if ENABLE(APPLE_PAY_INSTALLMENTS)
    ApplePayInstallmentConfiguration installmentConfiguration() const { return m_installmentConfiguration; }
#endif

#if ENABLE(APPLE_PAY_COUPON_CODE)
    std::optional<bool> supportsCouponCode() const { return m_supportsCouponCode; }
    const String& couponCode() const { return m_couponCode; }
#endif

#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
    std::optional<ApplePayShippingContactEditingMode> shippingContactEditingMode() const { return m_shippingContactEditingMode; }
#endif

#if ENABLE(APPLE_PAY_RECURRING_PAYMENTS)
    const std::optional<ApplePayRecurringPaymentRequest>& recurringPaymentRequest() const { return m_recurringPaymentRequest; }
#endif

#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS)
    const std::optional<ApplePayAutomaticReloadPaymentRequest>& automaticReloadPaymentRequest() const { return m_automaticReloadPaymentRequest; }
#endif

#if ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS)
    const std::optional<Vector<ApplePayPaymentTokenContext>>& multiTokenContexts() const { return m_multiTokenContexts; }
#endif

#if ENABLE(APPLE_PAY_DEFERRED_PAYMENTS)
    const std::optional<ApplePayDeferredPaymentRequest>& deferredPaymentRequest() const { return m_deferredPaymentRequest; }
#endif

#if ENABLE(APPLE_PAY_DISBURSEMENTS)
    const std::optional<ApplePayDisbursementRequest>& disbursementRequest() const { return m_disbursementRequest; }
#endif

#if ENABLE(APPLE_PAY_LATER_AVAILABILITY)
    const std::optional<ApplePayLaterAvailability> applePayLaterAvailability() const { return m_applePayLaterAvailability; }
#endif

#if ENABLE(APPLE_PAY_MERCHANT_CATEGORY_CODE)
    const String& merchantCategoryCode() const { return m_merchantCategoryCode; }
#endif

    bool installmentConfigurationReturnsNil() const;

private:
    explicit MockPaymentCoordinator(Page&);

    std::optional<String> validatedPaymentNetwork(const String&) const final;
    bool canMakePayments() final;
    void canMakePaymentsWithActiveCard(const String&, const String&, CompletionHandler<void(bool)>&&) final;
    void openPaymentSetup(const String&, const String&, CompletionHandler<void(bool)>&&) final;
    bool showPaymentUI(const URL&, const Vector<URL>&, const ApplePaySessionPaymentRequest&) final;
    void completeMerchantValidation(const PaymentMerchantSession&) final;
    void completeShippingMethodSelection(std::optional<ApplePayShippingMethodUpdate>&&) final;
    void completeShippingContactSelection(std::optional<ApplePayShippingContactUpdate>&&) final;
    void completePaymentMethodSelection(std::optional<ApplePayPaymentMethodUpdate>&&) final;
#if ENABLE(APPLE_PAY_COUPON_CODE)
    void completeCouponCodeChange(std::optional<ApplePayCouponCodeUpdate>&&) final;
#endif
    void completePaymentSession(ApplePayPaymentAuthorizationResult&&) final;
    void abortPaymentSession() final;
    void cancelPaymentSession() final;

    bool isMockPaymentCoordinator() const final { return true; }

    void getSetupFeatures(const ApplePaySetupConfiguration&, const URL&, CompletionHandler<void(Vector<Ref<ApplePaySetupFeature>>&&)>&&) final;
    void beginApplePaySetup(const ApplePaySetupConfiguration&, const URL&, Vector<Ref<ApplePaySetupFeature>>&&, CompletionHandler<void(bool)>&&) final;

    void dispatchIfShowing(Function<void()>&&);

    WeakPtr<PaymentCoordinator> m_paymentCoordinator;
    WeakPtr<Page> m_page;
    uint64_t m_showCount { 0 };
    uint64_t m_hideCount { 0 };
    bool m_canMakePayments { true };
    bool m_canMakePaymentsWithActiveCard { true };
    ApplePayPaymentContact m_shippingAddress;
    ApplePayLineItem m_total;
    Vector<ApplePayLineItem> m_lineItems;
    Vector<MockPaymentError> m_errors;
    Vector<ApplePayShippingMethod> m_shippingMethods;
    Vector<String> m_supportedCountries;
    HashSet<String, ASCIICaseInsensitiveHash> m_availablePaymentNetworks;
    MockPaymentContactFields m_requiredBillingContactFields;
    MockPaymentContactFields m_requiredShippingContactFields;
#if ENABLE(APPLE_PAY_INSTALLMENTS)
    ApplePayInstallmentConfiguration m_installmentConfiguration;
#endif
#if ENABLE(APPLE_PAY_COUPON_CODE)
    std::optional<bool> m_supportsCouponCode;
    String m_couponCode;
#endif
#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
    std::optional<ApplePayShippingContactEditingMode> m_shippingContactEditingMode;
#endif
    ApplePaySetupConfiguration m_setupConfiguration;
    Vector<Ref<ApplePaySetupFeature>> m_setupFeatures;

#if ENABLE(APPLE_PAY_RECURRING_PAYMENTS)
    std::optional<ApplePayRecurringPaymentRequest> m_recurringPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS)
    std::optional<ApplePayAutomaticReloadPaymentRequest> m_automaticReloadPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS)
    std::optional<Vector<ApplePayPaymentTokenContext>> m_multiTokenContexts;
#endif

#if ENABLE(APPLE_PAY_DEFERRED_PAYMENTS)
    std::optional<ApplePayDeferredPaymentRequest> m_deferredPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_DISBURSEMENTS)
    std::optional<ApplePayDisbursementRequest> m_disbursementRequest;
#endif

#if ENABLE(APPLE_PAY_LATER_AVAILABILITY)
    std::optional<ApplePayLaterAvailability> m_applePayLaterAvailability;
#endif

#if ENABLE(APPLE_PAY_MERCHANT_CATEGORY_CODE)
    String m_merchantCategoryCode;
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MockPaymentCoordinator)
    static bool isType(const WebCore::PaymentCoordinatorClient& paymentCoordinatorClient) { return paymentCoordinatorClient.isMockPaymentCoordinator(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(APPLE_PAY)
