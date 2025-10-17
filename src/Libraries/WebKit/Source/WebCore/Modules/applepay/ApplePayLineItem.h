/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

#include "ApplePayPaymentTiming.h"
#include "ApplePayRecurringPaymentDateUnit.h"
#include <optional>
#include <wtf/WallTime.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct ApplePayLineItem final {
    enum class Type : bool {
        Pending,
        Final,
    };

    Type type { Type::Final };
    String label;
    String amount;

    ApplePayPaymentTiming paymentTiming { ApplePayPaymentTiming::Immediate };

#if ENABLE(APPLE_PAY_RECURRING_LINE_ITEM)
    WallTime recurringPaymentStartDate { WallTime::nan() };
    ApplePayRecurringPaymentDateUnit recurringPaymentIntervalUnit { ApplePayRecurringPaymentDateUnit::Month };
    unsigned recurringPaymentIntervalCount = 1;
    WallTime recurringPaymentEndDate { WallTime::nan() };
#endif

#if ENABLE(APPLE_PAY_DEFERRED_LINE_ITEM)
    WallTime deferredPaymentDate { WallTime::nan() };
#endif

#if ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_LINE_ITEM)
    String automaticReloadPaymentThresholdAmount; /* required */
#endif

#if ENABLE(APPLE_PAY_DISBURSEMENTS)

    enum class DisbursementLineItemType : uint8_t {
        Disbursement,
        InstantFundsOutFee,
    };

    std::optional<DisbursementLineItemType> disbursementLineItemType;

#endif

};

} // namespace WebCore

#endif
