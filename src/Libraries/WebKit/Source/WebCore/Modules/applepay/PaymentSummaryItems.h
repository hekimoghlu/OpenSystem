/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

OBJC_CLASS NSArray;
OBJC_CLASS NSDecimalNumber;
OBJC_CLASS PKAutomaticReloadPaymentSummaryItem;
OBJC_CLASS PKDeferredPaymentSummaryItem;
OBJC_CLASS PKPaymentSummaryItem;
OBJC_CLASS PKRecurringPaymentSummaryItem;

#if HAVE(PASSKIT_DISBURSEMENTS)
OBJC_CLASS PKDisbursementSummaryItem;
OBJC_CLASS PKInstantFundsOutFeeSummaryItem;
#endif // HAVE(PASSKIT_DISBURSEMENTS)

namespace WebCore {

struct ApplePayLineItem;

#if HAVE(PASSKIT_RECURRING_SUMMARY_ITEM)
WEBCORE_EXPORT PKRecurringPaymentSummaryItem *platformRecurringSummaryItem(const ApplePayLineItem&);
#endif

#if HAVE(PASSKIT_DEFERRED_SUMMARY_ITEM)
WEBCORE_EXPORT PKDeferredPaymentSummaryItem *platformDeferredSummaryItem(const ApplePayLineItem&);
#endif

#if HAVE(PASSKIT_AUTOMATIC_RELOAD_SUMMARY_ITEM)
WEBCORE_EXPORT PKAutomaticReloadPaymentSummaryItem *platformAutomaticReloadSummaryItem(const ApplePayLineItem&);
#endif

#if HAVE(PASSKIT_DISBURSEMENTS)
WEBCORE_EXPORT PKDisbursementSummaryItem *platformDisbursementSummaryItem(const ApplePayLineItem&);
WEBCORE_EXPORT PKInstantFundsOutFeeSummaryItem *platformInstantFundsOutFeeSummaryItem(const ApplePayLineItem&);
#endif // HAVE(PASSKIT_DISBURSEMENTS)

WEBCORE_EXPORT PKPaymentSummaryItem *platformSummaryItem(const ApplePayLineItem&);
WEBCORE_EXPORT NSArray *platformDisbursementSummaryItems(const Vector<ApplePayLineItem>&);
WEBCORE_EXPORT NSArray *platformSummaryItems(const ApplePayLineItem& total, const Vector<ApplePayLineItem>&);

WEBCORE_EXPORT NSDecimalNumber *toDecimalNumber(const String& amount);

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
