/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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

#if ENABLE(APPLE_PAY) && ENABLE(PAYMENT_REQUEST)

#include "ApplePayAutomaticReloadPaymentRequest.h"
#include "ApplePayDeferredPaymentRequest.h"
#include "ApplePayDisbursementRequest.h"
#include "ApplePayLineItem.h"
#include "ApplePayPaymentMethodType.h"
#include "ApplePayPaymentTokenContext.h"
#include "ApplePayRecurringPaymentRequest.h"
#include "ApplePayShippingMethod.h"

namespace WebCore {

struct ApplePayModifier {
    std::optional<ApplePayPaymentMethodType> paymentMethodType;
    std::optional<ApplePayLineItem> total;
    Vector<ApplePayLineItem> additionalLineItems;
#if ENABLE(APPLE_PAY_UPDATE_SHIPPING_METHODS_WHEN_CHANGING_LINE_ITEMS)
    Vector<ApplePayShippingMethod> additionalShippingMethods;
#endif

#if ENABLE(APPLE_PAY_RECURRING_PAYMENTS)
    std::optional<ApplePayRecurringPaymentRequest> recurringPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS)
    std::optional<ApplePayAutomaticReloadPaymentRequest> automaticReloadPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS)
    std::optional<Vector<ApplePayPaymentTokenContext>> multiTokenContexts;
#endif

#if ENABLE(APPLE_PAY_DEFERRED_PAYMENTS)
    std::optional<ApplePayDeferredPaymentRequest> deferredPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_DISBURSEMENTS)
    std::optional<ApplePayDisbursementRequest> disbursementRequest;
#endif
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY) && ENABLE(PAYMENT_REQUEST)
