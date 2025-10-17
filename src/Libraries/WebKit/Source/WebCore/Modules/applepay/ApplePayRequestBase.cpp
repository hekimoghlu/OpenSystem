/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#include "config.h"
#include "ApplePayRequestBase.h"

#if ENABLE(APPLE_PAY)

#include "PaymentCoordinator.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

static bool requiresSupportedNetworks(unsigned version, const ApplePayRequestBase& request)
{
#if ENABLE(APPLE_PAY_INSTALLMENTS)
    return version < 8 || !request.installmentConfiguration;
#else
    UNUSED_PARAM(version);
    UNUSED_PARAM(request);
    return true;
#endif
}

static ExceptionOr<Vector<String>> convertAndValidate(Document& document, unsigned version, const Vector<String>& supportedNetworks, const PaymentCoordinator& paymentCoordinator)
{
    Vector<String> result;
    result.reserveInitialCapacity(supportedNetworks.size());
    for (auto& supportedNetwork : supportedNetworks) {
        auto validatedNetwork = paymentCoordinator.validatedPaymentNetwork(document, version, supportedNetwork);
        if (!validatedNetwork)
            return Exception { ExceptionCode::TypeError, makeString("\""_s, supportedNetwork, "\" is not a valid payment network."_s) };
        result.append(*validatedNetwork);
    }

    return WTFMove(result);
}

ExceptionOr<ApplePaySessionPaymentRequest> convertAndValidate(Document& document, unsigned version, const ApplePayRequestBase& request, const PaymentCoordinator& paymentCoordinator)
{
    if (!version || !paymentCoordinator.supportsVersion(document, version))
        return Exception { ExceptionCode::InvalidAccessError, makeString('"', version, "\" is not a supported version."_s) };

    ApplePaySessionPaymentRequest result;
    result.setVersion(version);
    result.setCountryCode(request.countryCode);

    auto merchantCapabilities = convertAndValidate(request.merchantCapabilities);
    if (merchantCapabilities.hasException())
        return merchantCapabilities.releaseException();
    result.setMerchantCapabilities(merchantCapabilities.releaseReturnValue());

    if (requiresSupportedNetworks(version, request) && request.supportedNetworks.isEmpty())
        return Exception { ExceptionCode::TypeError, "At least one supported network must be provided."_s };

    auto supportedNetworks = convertAndValidate(document, version, request.supportedNetworks, paymentCoordinator);
    if (supportedNetworks.hasException())
        return supportedNetworks.releaseException();
    result.setSupportedNetworks(supportedNetworks.releaseReturnValue());

    if (request.requiredBillingContactFields) {
        auto requiredBillingContactFields = convertAndValidate(version, *request.requiredBillingContactFields);
        if (requiredBillingContactFields.hasException())
            return requiredBillingContactFields.releaseException();
        result.setRequiredBillingContactFields(requiredBillingContactFields.releaseReturnValue());
    }

    if (request.billingContact)
        result.setBillingContact(PaymentContact::fromApplePayPaymentContact(version, *request.billingContact));

    if (request.requiredShippingContactFields) {
        auto requiredShippingContactFields = convertAndValidate(version, *request.requiredShippingContactFields);
        if (requiredShippingContactFields.hasException())
            return requiredShippingContactFields.releaseException();
        result.setRequiredShippingContactFields(requiredShippingContactFields.releaseReturnValue());
    }

    if (request.shippingContact)
        result.setShippingContact(PaymentContact::fromApplePayPaymentContact(version, *request.shippingContact));

    result.setApplicationData(request.applicationData);

    if (version >= 3)
        result.setSupportedCountries(Vector { request.supportedCountries });

#if ENABLE(APPLE_PAY_INSTALLMENTS)
    if (request.installmentConfiguration) {
        auto installmentConfiguration = PaymentInstallmentConfiguration::create(*request.installmentConfiguration);
        if (installmentConfiguration.hasException())
            return installmentConfiguration.releaseException();
        result.setInstallmentConfiguration(installmentConfiguration.releaseReturnValue());
    }
#endif

#if ENABLE(APPLE_PAY_COUPON_CODE)
    result.setSupportsCouponCode(request.supportsCouponCode);
    result.setCouponCode(request.couponCode);
#endif

#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
    result.setShippingContactEditingMode(request.shippingContactEditingMode);
#endif

#if ENABLE(APPLE_PAY_LATER_AVAILABILITY)
    result.setApplePayLaterAvailability(request.applePayLaterAvailability);
#endif

#if ENABLE(APPLE_PAY_MERCHANT_CATEGORY_CODE)
    result.setMerchantCategoryCode(request.merchantCategoryCode);
#endif

    return WTFMove(result);
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
