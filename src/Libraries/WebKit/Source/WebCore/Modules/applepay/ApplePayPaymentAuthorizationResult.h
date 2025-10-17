/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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

#include "ApplePayError.h"
#include "ApplePayPaymentOrderDetails.h"
#include <optional>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

struct ApplePayPaymentAuthorizationResult {
    using Status = unsigned short;
    static constexpr Status Success = 0;
    static constexpr Status Failure = 1;
    static constexpr Status InvalidBillingPostalAddress = 2;
    static constexpr Status InvalidShippingPostalAddress = 3;
    static constexpr Status InvalidShippingContact = 4;
    static constexpr Status PINRequired = 5;
    static constexpr Status PINIncorrect = 6;
    static constexpr Status PINLockout = 7;

    Status status; // required
    Vector<Ref<ApplePayError>> errors;

#if ENABLE(APPLE_PAY_PAYMENT_ORDER_DETAILS)
    std::optional<ApplePayPaymentOrderDetails> orderDetails;
#endif

    WEBCORE_EXPORT bool isFinalState() const;
};

}

#endif
