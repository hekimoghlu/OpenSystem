/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#include "ApplePayLineItem.h"
#include "ApplePayPaymentTokenContext.h"
#include "ApplePayRecurringPaymentRequest.h"
#include <wtf/Vector.h>

namespace WebCore {

struct ApplePayDetailsUpdateBase {
    ApplePayLineItem newTotal;
    Vector<ApplePayLineItem> newLineItems;

#if ENABLE(APPLE_PAY_RECURRING_PAYMENTS)
    std::optional<ApplePayRecurringPaymentRequest> newRecurringPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS)
    std::optional<ApplePayAutomaticReloadPaymentRequest> newAutomaticReloadPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS)
    std::optional<Vector<ApplePayPaymentTokenContext>> newMultiTokenContexts;
#endif

#if ENABLE(APPLE_PAY_DEFERRED_PAYMENTS)
    std::optional<ApplePayDeferredPaymentRequest> newDeferredPaymentRequest;
#endif

#if ENABLE(APPLE_PAY_DISBURSEMENTS)
    std::optional<ApplePayDisbursementRequest> newDisbursementRequest;
#endif
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
