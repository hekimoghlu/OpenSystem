/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

#include "ApplePayAutomaticReloadPaymentRequest.h"
#include "ApplePayDeferredPaymentRequest.h"
#include "ApplePayDisbursementRequest.h"
#include "ApplePayError.h"
#include "ApplePayLaterAvailability.h"
#include "ApplePayLineItem.h"
#include "ApplePayPaymentTokenContext.h"
#include "ApplePayRecurringPaymentRequest.h"
#include "ApplePayShippingContactEditingMode.h"
#include "ApplePayShippingMethod.h"
#include "PaymentContact.h"
#include "PaymentInstallmentConfigurationWebCore.h"
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class ApplePaySessionPaymentRequestShippingType : uint8_t {
    Shipping,
    Delivery,
    StorePickup,
    ServicePickup,
};

struct ApplePaySessionPaymentRequestContactFields {
    bool postalAddress { false };
    bool phone { false };
    bool email { false };
    bool name { false };
    bool phoneticName { false };
};

class ApplePaySessionPaymentRequest {
public:
    WEBCORE_EXPORT ApplePaySessionPaymentRequest();
    WEBCORE_EXPORT ~ApplePaySessionPaymentRequest();

    unsigned version() const { return m_version; }
    void setVersion(unsigned version) { m_version = version; }

    const String& countryCode() const { return m_countryCode; }
    void setCountryCode(const String& countryCode) { m_countryCode = countryCode; }

    const String& currencyCode() const { return m_currencyCode; }
    void setCurrencyCode(const String& currencyCode) { m_currencyCode = currencyCode; }

    using ContactFields = ApplePaySessionPaymentRequestContactFields;

    const ContactFields& requiredBillingContactFields() const { return m_requiredBillingContactFields; }
    void setRequiredBillingContactFields(const ContactFields& requiredBillingContactFields) { m_requiredBillingContactFields = requiredBillingContactFields; }

    const PaymentContact& billingContact() const { return m_billingContact; }
    void setBillingContact(const PaymentContact& billingContact) { m_billingContact = billingContact; }

    const ContactFields& requiredShippingContactFields() const { return m_requiredShippingContactFields; }
    void setRequiredShippingContactFields(const ContactFields& requiredShippingContactFields) { m_requiredShippingContactFields = requiredShippingContactFields; }

    const PaymentContact& shippingContact() const { return m_shippingContact; }
    void setShippingContact(const PaymentContact& shippingContact) { m_shippingContact = shippingContact; }

    const Vector<String>& supportedNetworks() const { return m_supportedNetworks; }
    void setSupportedNetworks(const Vector<String>& supportedNetworks) { m_supportedNetworks = supportedNetworks; }

    struct MerchantCapabilities {
        bool supports3DS { false };
        bool supportsEMV { false };
        bool supportsCredit { false };
        bool supportsDebit { false };
#if ENABLE(APPLE_PAY_DISBURSEMENTS)
        bool supportsInstantFundsOut { false };
#endif
    };

    const MerchantCapabilities& merchantCapabilities() const { return m_merchantCapabilities; }
    void setMerchantCapabilities(const MerchantCapabilities& merchantCapabilities) { m_merchantCapabilities = merchantCapabilities; }

    using ShippingType = ApplePaySessionPaymentRequestShippingType;
    ShippingType shippingType() const { return m_shippingType; }
    void setShippingType(ShippingType shippingType) { m_shippingType = shippingType; }

    const Vector<ApplePayShippingMethod>& shippingMethods() const { return m_shippingMethods; }
    void setShippingMethods(const Vector<ApplePayShippingMethod>& shippingMethods) { m_shippingMethods = shippingMethods; }

    const Vector<ApplePayLineItem>& lineItems() const { return m_lineItems; }
    void setLineItems(const Vector<ApplePayLineItem>& lineItems) { m_lineItems = lineItems; }

    const ApplePayLineItem& total() const { return m_total; };
    void setTotal(const ApplePayLineItem& total) { m_total = total; }

    const String& applicationData() const { return m_applicationData; }
    void setApplicationData(const String& applicationData) { m_applicationData = applicationData; }

    const Vector<String>& supportedCountries() const { return m_supportedCountries; }
    void setSupportedCountries(Vector<String>&& supportedCountries) { m_supportedCountries = WTFMove(supportedCountries); }

    enum class Requester : bool {
        ApplePayJS,
        PaymentRequest,
    };

    Requester requester() const { return m_requester; }
    void setRequester(Requester requester) { m_requester = requester; }

#if HAVE(PASSKIT_INSTALLMENTS)
    const PaymentInstallmentConfiguration& installmentConfiguration() const { return m_installmentConfiguration; }
    void setInstallmentConfiguration(PaymentInstallmentConfiguration&& installmentConfiguration) { m_installmentConfiguration = WTFMove(installmentConfiguration); }
#endif

#if ENABLE(APPLE_PAY_COUPON_CODE)
    std::optional<bool> supportsCouponCode() const { return m_supportsCouponCode; }
    void setSupportsCouponCode(std::optional<bool> supportsCouponCode) { m_supportsCouponCode = supportsCouponCode; }

    const String& couponCode() const { return m_couponCode; }
    void setCouponCode(const String& couponCode) { m_couponCode = couponCode; }
#endif

#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
    const std::optional<ApplePayShippingContactEditingMode>& shippingContactEditingMode() const { return m_shippingContactEditingMode; }
    void setShippingContactEditingMode(const std::optional<ApplePayShippingContactEditingMode>& shippingContactEditingMode) { m_shippingContactEditingMode = shippingContactEditingMode; }
#endif

#if ENABLE(APPLE_PAY_RECURRING_PAYMENTS)
    const std::optional<ApplePayRecurringPaymentRequest>& recurringPaymentRequest() const { return m_recurringPaymentRequest; }
    void setRecurringPaymentRequest(std::optional<ApplePayRecurringPaymentRequest>&& recurringPaymentRequest) { m_recurringPaymentRequest = WTFMove(recurringPaymentRequest); }
#endif

#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS)
    const std::optional<ApplePayAutomaticReloadPaymentRequest>& automaticReloadPaymentRequest() const { return m_automaticReloadPaymentRequest; }
    void setAutomaticReloadPaymentRequest(std::optional<ApplePayAutomaticReloadPaymentRequest>&& automaticReloadPaymentRequest) { m_automaticReloadPaymentRequest = WTFMove(automaticReloadPaymentRequest); }
#endif

#if ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS)
    const std::optional<Vector<ApplePayPaymentTokenContext>>& multiTokenContexts() const { return m_multiTokenContexts; }
    void setMultiTokenContexts(std::optional<Vector<ApplePayPaymentTokenContext>>&& multiTokenContexts) { m_multiTokenContexts = WTFMove(multiTokenContexts); }
#endif

#if ENABLE(APPLE_PAY_DEFERRED_PAYMENTS)
    const std::optional<ApplePayDeferredPaymentRequest>& deferredPaymentRequest() const { return m_deferredPaymentRequest; }
    void setDeferredPaymentRequest(std::optional<ApplePayDeferredPaymentRequest>&& deferredPaymentRequest) { m_deferredPaymentRequest = WTFMove(deferredPaymentRequest); }
#endif

#if ENABLE(APPLE_PAY_DISBURSEMENTS)
    const std::optional<ApplePayDisbursementRequest>& disbursementRequest() const { return m_disbursementRequest; }
    void setDisbursementRequest(std::optional<ApplePayDisbursementRequest>&& disbursementRequest) { m_disbursementRequest = WTFMove(disbursementRequest); }
#endif

#if ENABLE(APPLE_PAY_LATER_AVAILABILITY)
    const std::optional<ApplePayLaterAvailability>& applePayLaterAvailability() const { return m_applePayLaterAvailability; }
    void setApplePayLaterAvailability(const std::optional<ApplePayLaterAvailability>& applePayLaterAvailability) { m_applePayLaterAvailability = applePayLaterAvailability; }
#endif

#if ENABLE(APPLE_PAY_MERCHANT_CATEGORY_CODE)
    const String& merchantCategoryCode() const { return m_merchantCategoryCode; }
    void setMerchantCategoryCode(const String& merchantCategoryCode) { m_merchantCategoryCode = merchantCategoryCode; }
#endif

    ApplePaySessionPaymentRequest(String&& countryCode
        , String&& currencyCode
        , ContactFields&& requiredBillingContactFields
        , PaymentContact&& billingContact
        , ContactFields&& requiredShippingContactFields
        , PaymentContact&& shippingContact
        , Vector<String>&& supportedNetworks
        , MerchantCapabilities&& merchantCapabilities
        , ApplePaySessionPaymentRequestShippingType&& shippingType
        , Vector<ApplePayShippingMethod>&& shippingMethods
        , Vector<ApplePayLineItem>&& lineItems
        , ApplePayLineItem&& total
        , String applicationData
        , Vector<String>&& supportedCountries
        , Requester&& requester
#if HAVE(PASSKIT_INSTALLMENTS)
        , PaymentInstallmentConfiguration&& installmentConfiguration
#endif
#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
        , std::optional<ApplePayShippingContactEditingMode>&& shippingContactEditingMode
#endif
#if ENABLE(APPLE_PAY_COUPON_CODE)
        , std::optional<bool> supportsCouponCode
        , String couponCode
#endif
#if ENABLE(APPLE_PAY_RECURRING_PAYMENTS)
        , std::optional<ApplePayRecurringPaymentRequest>&& recurringPaymentRequest
#endif
#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS)
        , std::optional<ApplePayAutomaticReloadPaymentRequest>&& automaticReloadPaymentRequest
#endif
#if ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS)
        , std::optional<Vector<ApplePayPaymentTokenContext>>&& multiTokenContexts
#endif
#if ENABLE(APPLE_PAY_DEFERRED_PAYMENTS)
        , std::optional<ApplePayDeferredPaymentRequest>&& deferredPaymentRequest
#endif
#if ENABLE(APPLE_PAY_DISBURSEMENTS)
        , std::optional<ApplePayDisbursementRequest>&& disbursementRequest
#endif
#if ENABLE(APPLE_PAY_LATER_AVAILABILITY)
        , std::optional<ApplePayLaterAvailability>&& applePayLaterAvailability
#endif
#if ENABLE(APPLE_PAY_MERCHANT_CATEGORY_CODE)
        , String&& merchantCategoryCode
#endif
        )
            : m_countryCode(WTFMove(countryCode))
            , m_currencyCode(WTFMove(currencyCode))
            , m_requiredBillingContactFields(WTFMove(requiredBillingContactFields))
            , m_billingContact(WTFMove(billingContact))
            , m_requiredShippingContactFields(WTFMove(requiredShippingContactFields))
            , m_shippingContact(WTFMove(shippingContact))
            , m_supportedNetworks(WTFMove(supportedNetworks))
            , m_merchantCapabilities(WTFMove(merchantCapabilities))
            , m_shippingType(WTFMove(shippingType))
            , m_shippingMethods(WTFMove(shippingMethods))
            , m_lineItems(WTFMove(lineItems))
            , m_total(WTFMove(total))
            , m_applicationData(WTFMove(applicationData))
            , m_supportedCountries(WTFMove(supportedCountries))
            , m_requester(WTFMove(requester))
#if HAVE(PASSKIT_INSTALLMENTS)
            , m_installmentConfiguration(WTFMove(installmentConfiguration))
#endif
#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
            , m_shippingContactEditingMode(WTFMove(shippingContactEditingMode))
#endif
#if ENABLE(APPLE_PAY_COUPON_CODE)
            , m_supportsCouponCode(WTFMove(supportsCouponCode))
            , m_couponCode(WTFMove(couponCode))
#endif
#if ENABLE(APPLE_PAY_RECURRING_PAYMENTS)
            , m_recurringPaymentRequest(WTFMove(recurringPaymentRequest))
#endif
#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS)
            , m_automaticReloadPaymentRequest(WTFMove(automaticReloadPaymentRequest))
#endif
#if ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS)
            , m_multiTokenContexts(WTFMove(multiTokenContexts))
#endif
#if ENABLE(APPLE_PAY_DEFERRED_PAYMENTS)
            , m_deferredPaymentRequest(WTFMove(deferredPaymentRequest))
#endif
#if ENABLE(APPLE_PAY_DISBURSEMENTS)
            , m_disbursementRequest(WTFMove(disbursementRequest))
#endif
#if ENABLE(APPLE_PAY_LATER_AVAILABILITY)
            , m_applePayLaterAvailability(WTFMove(applePayLaterAvailability))
#endif
#if ENABLE(APPLE_PAY_MERCHANT_CATEGORY_CODE)
            , m_merchantCategoryCode(WTFMove(merchantCategoryCode))
#endif
            { }

private:
    unsigned m_version { 0 };

    String m_countryCode;
    String m_currencyCode;

    ContactFields m_requiredBillingContactFields;
    PaymentContact m_billingContact;

    ContactFields m_requiredShippingContactFields;
    PaymentContact m_shippingContact;

    Vector<String> m_supportedNetworks;
    MerchantCapabilities m_merchantCapabilities;

    ShippingType m_shippingType { ShippingType::Shipping };
    Vector<ApplePayShippingMethod> m_shippingMethods;

    Vector<ApplePayLineItem> m_lineItems;
    ApplePayLineItem m_total;

    String m_applicationData;
    Vector<String> m_supportedCountries;

    Requester m_requester { Requester::ApplePayJS };

#if HAVE(PASSKIT_INSTALLMENTS)
    PaymentInstallmentConfiguration m_installmentConfiguration;
#endif

#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
    std::optional<ApplePayShippingContactEditingMode> m_shippingContactEditingMode;
#endif

#if ENABLE(APPLE_PAY_COUPON_CODE)
    std::optional<bool> m_supportsCouponCode;
    String m_couponCode;
#endif

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

#endif
