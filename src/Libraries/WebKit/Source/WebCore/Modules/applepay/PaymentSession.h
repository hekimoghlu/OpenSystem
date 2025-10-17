/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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

#include "ExceptionOr.h"
#include "PaymentSessionBase.h"

namespace WebCore {

class Document;
class Payment;
class PaymentContact;
class PaymentMethod;
class PaymentSessionError;
class ScriptExecutionContext;
struct ApplePayShippingMethod;

class PaymentSession : public virtual PaymentSessionBase {
public:
    static ExceptionOr<void> canCreateSession(Document&);

    virtual unsigned version() const = 0;
    virtual void validateMerchant(URL&&) = 0;
    virtual void didAuthorizePayment(const Payment&) = 0;
    virtual void didSelectShippingMethod(const ApplePayShippingMethod&) = 0;
    virtual void didSelectShippingContact(const PaymentContact&) = 0;
    virtual void didSelectPaymentMethod(const PaymentMethod&) = 0;
#if ENABLE(APPLE_PAY_COUPON_CODE)
    virtual void didChangeCouponCode(String&& couponCode) = 0;
#endif
    virtual void didCancelPaymentSession(PaymentSessionError&&) = 0;
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
