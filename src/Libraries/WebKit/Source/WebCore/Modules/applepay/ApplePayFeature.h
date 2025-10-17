/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#include <wtf/Forward.h>

namespace WebCore {

enum class ApplePayFeature : uint8_t {
#if ENABLE(APPLE_PAY_LATER)
    ApplePayLater,
#endif
#if ENABLE(APPLE_PAY_LATER_AVAILABILITY)
    ApplePayLaterAvailability,
#endif
#if ENABLE(APPLE_PAY_PAYMENT_ORDER_DETAILS)
    AuthorizationResultOrderDetails,
#endif
    LineItemPaymentTiming,
#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS)
    PaymentRequestAutomaticReload,
#endif
#if ENABLE(APPLE_PAY_COUPON_CODE)
    PaymentRequestCouponCode,
#endif
#if ENABLE(APPLE_PAY_MERCHANT_CATEGORY_CODE)
    PaymentRequestMerchantCategoryCode,
#endif
#if ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS)
    PaymentRequestMultiTokenContexts,
#endif
#if ENABLE(APPLE_PAY_RECURRING_PAYMENTS)
    PaymentRequestRecurring,
#endif
#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
    PaymentRequestShippingContactEditingMode,
#endif
#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_LINE_ITEM)
    PaymentTimingAutomaticReload,
#endif
#if ENABLE(APPLE_PAY_DEFERRED_PAYMENTS)
    PaymentRequestDeferred,
#endif
#if ENABLE(APPLE_PAY_DEFERRED_LINE_ITEM)
    PaymentTimingDeferred,
#endif
    PaymentTimingImmediate,
#if ENABLE(APPLE_PAY_RECURRING_LINE_ITEM)
    PaymentTimingRecurring,
#endif
#if ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE)
    ShippingContactEditingModeEnabled,
    ShippingContactEditingModeStorePickup,
#endif
#if ENABLE(APPLE_PAY_SHIPPING_METHOD_DATE_COMPONENTS_RANGE)
    ShippingMethodDateComponentsRange,
#endif
#if ENABLE(APPLE_PAY_DISBURSEMENTS)
    PaymentRequestDisbursements,
#endif
};

} // namespace WebCore

