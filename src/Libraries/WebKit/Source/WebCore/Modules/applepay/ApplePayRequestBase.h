/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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

#include "ApplePayContactField.h"
#include "ApplePayFeature.h"
#include "ApplePayInstallmentConfigurationWebCore.h"
#include "ApplePayMerchantCapability.h"
#include "ApplePayPaymentContact.h"
#include "ApplePayShippingContactEditingMode.h"

namespace WebCore {

class Document;
class PaymentCoordinator;

struct ApplePayRequestBase {
    std::optional<Vector<ApplePayFeature>> features;

    Vector<ApplePayMerchantCapability> merchantCapabilities;
    Vector<String> supportedNetworks;
    String countryCode;

    std::optional<Vector<ApplePayContactField>> requiredBillingContactFields;
    std::optional<ApplePayPaymentContact> billingContact;

    std::optional<Vector<ApplePayContactField>> requiredShippingContactFields;
    std::optional<ApplePayPaymentContact> shippingContact;

    String applicationData;
    Vector<String> supportedCountries;

#if ENABLE(APPLE_PAY_INSTALLMENTS)
    std::optional<ApplePayInstallmentConfiguration> installmentConfiguration;
#endif

#if ENABLE(APPLE_PAY_COUPON_CODE)
    std::optional<bool> supportsCouponCode;
    String couponCode;
#endif

#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
    std::optional<ApplePayShippingContactEditingMode> shippingContactEditingMode;
#endif

#if ENABLE(APPLE_PAY_LATER_AVAILABILITY)
    std::optional<ApplePayLaterAvailability> applePayLaterAvailability;
#endif

#if ENABLE(APPLE_PAY_MERCHANT_CATEGORY_CODE)
    String merchantCategoryCode;
#endif
};

ExceptionOr<ApplePaySessionPaymentRequest> convertAndValidate(Document&, unsigned version, const ApplePayRequestBase&, const PaymentCoordinator&);

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
