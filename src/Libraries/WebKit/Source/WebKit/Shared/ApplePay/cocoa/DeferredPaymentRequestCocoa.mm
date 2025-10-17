/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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
#import "DeferredPaymentRequest.h"

#if HAVE(PASSKIT_DEFERRED_PAYMENTS)

#import <WebCore/ApplePayDeferredPaymentRequest.h>
#import <WebCore/PaymentSummaryItems.h>
#import <wtf/RetainPtr.h>

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebKit {
using namespace WebCore;

RetainPtr<PKDeferredPaymentRequest> platformDeferredPaymentRequest(const ApplePayDeferredPaymentRequest& webDeferredPaymentRequest)
{
    auto pkDeferredPaymentRequest = adoptNS([PAL::allocPKDeferredPaymentRequestInstance()
        initWithPaymentDescription:webDeferredPaymentRequest.paymentDescription
        deferredBilling:platformDeferredSummaryItem(webDeferredPaymentRequest.deferredBilling)
        managementURL:[NSURL URLWithString:webDeferredPaymentRequest.managementURL]]);
    if (auto& billingAgreement = webDeferredPaymentRequest.billingAgreement; !billingAgreement.isNull())
        [pkDeferredPaymentRequest setBillingAgreement:billingAgreement];
    if (auto& freeCancellationDate = webDeferredPaymentRequest.freeCancellationDate; !freeCancellationDate.isNaN()) {
        if (auto& freeCancellationDateTimeZone = webDeferredPaymentRequest.freeCancellationDateTimeZone; !freeCancellationDateTimeZone.isNull()) {
            if (auto timeZone = [NSTimeZone timeZoneWithName:freeCancellationDateTimeZone]) {
                [pkDeferredPaymentRequest setFreeCancellationDate:[NSDate dateWithTimeIntervalSince1970:freeCancellationDate.secondsSinceEpoch().value()]];
                [pkDeferredPaymentRequest setFreeCancellationDateTimeZone:timeZone];
            }
        }
    }
    if (auto& tokenNotificationURL = webDeferredPaymentRequest.tokenNotificationURL; !tokenNotificationURL.isNull())
        [pkDeferredPaymentRequest setTokenNotificationURL:[NSURL URLWithString:tokenNotificationURL]];
    return pkDeferredPaymentRequest;
}

} // namespace WebKit

#endif // HAVE(PASSKIT_DEFERRED_PAYMENTS)
