/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#import "RecurringPaymentRequest.h"

#if HAVE(PASSKIT_RECURRING_PAYMENTS)

#import <WebCore/ApplePayRecurringPaymentRequest.h>
#import <WebCore/PaymentSummaryItems.h>
#import <wtf/RetainPtr.h>

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebKit {
using namespace WebCore;

RetainPtr<PKRecurringPaymentRequest> platformRecurringPaymentRequest(const ApplePayRecurringPaymentRequest& webRecurringPaymentRequest)
{
    auto pkRecurringPaymentRequest = adoptNS([PAL::allocPKRecurringPaymentRequestInstance() initWithPaymentDescription:webRecurringPaymentRequest.paymentDescription regularBilling:platformRecurringSummaryItem(webRecurringPaymentRequest.regularBilling) managementURL:[NSURL URLWithString:webRecurringPaymentRequest.managementURL]]);
    if (auto& trialBilling = webRecurringPaymentRequest.trialBilling)
        [pkRecurringPaymentRequest setTrialBilling:platformRecurringSummaryItem(*trialBilling)];
    if (auto& billingAgreement = webRecurringPaymentRequest.billingAgreement; !billingAgreement.isNull())
        [pkRecurringPaymentRequest setBillingAgreement:billingAgreement];
    if (auto& tokenNotificationURL = webRecurringPaymentRequest.tokenNotificationURL; !tokenNotificationURL.isNull())
        [pkRecurringPaymentRequest setTokenNotificationURL:[NSURL URLWithString:tokenNotificationURL]];
    return pkRecurringPaymentRequest;
}

} // namespace WebKit

#endif // HAVE(PASSKIT_RECURRING_PAYMENTS)
