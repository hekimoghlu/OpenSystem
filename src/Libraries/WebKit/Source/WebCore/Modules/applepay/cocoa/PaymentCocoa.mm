/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
#import "config.h"
#import "Payment.h"

#if ENABLE(APPLE_PAY)

#import "ApplePayPayment.h"
#import "PaymentContact.h"
#import "PaymentMethod.h"
#import <pal/spi/cocoa/PassKitSPI.h>
#import <wtf/cocoa/SpanCocoa.h>

namespace WebCore {

static void finishConverting(PKPayment *payment, ApplePayPayment& result)
{
#if HAVE(PASSKIT_INSTALLMENTS)
    if (NSString *installmentAuthorizationToken = payment.installmentAuthorizationToken)
        result.installmentAuthorizationToken = installmentAuthorizationToken;
#else
    UNUSED_PARAM(payment);
    UNUSED_PARAM(result);
#endif
}

static ApplePayPayment::Token convert(PKPaymentToken *paymentToken)
{
    ASSERT(paymentToken);

    ApplePayPayment::Token result;

    result.paymentMethod = PaymentMethod(paymentToken.paymentMethod).toApplePayPaymentMethod();

    if (RetainPtr<NSString> transactionIdentifier = paymentToken.transactionIdentifier)
        result.transactionIdentifier = transactionIdentifier.get();
    if (RetainPtr<NSData> paymentData = paymentToken.paymentData)
        result.paymentData = String::fromUTF8(span(paymentData.get()));

    return result;
}

static ApplePayPayment convert(unsigned version, PKPayment *payment)
{
    ASSERT(payment);

    ApplePayPayment result;

    result.token = convert(payment.token);

    if (payment.billingContact)
        result.billingContact = PaymentContact(payment.billingContact).toApplePayPaymentContact(version);
    if (payment.shippingContact)
        result.shippingContact = PaymentContact(payment.shippingContact).toApplePayPaymentContact(version);

    finishConverting(payment, result);

    return result;
}
    
Payment::Payment() = default;

Payment::Payment(RetainPtr<PKPayment>&& pkPayment)
    : m_pkPayment { WTFMove(pkPayment) }
{
}

Payment::~Payment() = default;

ApplePayPayment Payment::toApplePayPayment(unsigned version) const
{
    return convert(version, m_pkPayment.get());
}

RetainPtr<PKPayment> Payment::pkPayment() const
{
    return m_pkPayment;
}

}

#endif
