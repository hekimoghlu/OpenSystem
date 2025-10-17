/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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
#import "PaymentSummaryItems.h"

#if ENABLE(APPLE_PAY)

#import "ApplePayLineItem.h"
#import <pal/cocoa/PassKitSoftLink.h>

namespace WebCore {

NSDecimalNumber *toDecimalNumber(const String& amount)
{
    if (!amount)
        return [NSDecimalNumber zero];
    return [NSDecimalNumber decimalNumberWithString:amount locale:@{ NSLocaleDecimalSeparator : @"." }];
}

static PKPaymentSummaryItemType toPKPaymentSummaryItemType(ApplePayLineItem::Type type)
{
    switch (type) {
    case ApplePayLineItem::Type::Final:
        return PKPaymentSummaryItemTypeFinal;
    case ApplePayLineItem::Type::Pending:
        return PKPaymentSummaryItemTypePending;
    }
}

} // namespace WebCore

namespace WebCore {

#if HAVE(PASSKIT_RECURRING_SUMMARY_ITEM) || HAVE(PASSKIT_DEFERRED_SUMMARY_ITEM)

static NSDate *toDate(WallTime date)
{
    return [NSDate dateWithTimeIntervalSince1970:date.secondsSinceEpoch().value()];
}

#endif // HAVE(PASSKIT_RECURRING_SUMMARY_ITEM) || HAVE(PASSKIT_DEFERRED_SUMMARY_ITEM)

#if HAVE(PASSKIT_RECURRING_SUMMARY_ITEM)

static NSCalendarUnit toCalendarUnit(ApplePayRecurringPaymentDateUnit unit)
{
    switch (unit) {
    case ApplePayRecurringPaymentDateUnit::Year:
        return NSCalendarUnitYear;

    case ApplePayRecurringPaymentDateUnit::Month:
        return NSCalendarUnitMonth;

    case ApplePayRecurringPaymentDateUnit::Day:
        return NSCalendarUnitDay;

    case ApplePayRecurringPaymentDateUnit::Hour:
        return NSCalendarUnitHour;

    case ApplePayRecurringPaymentDateUnit::Minute:
        return NSCalendarUnitMinute;
    }
}

PKRecurringPaymentSummaryItem *platformRecurringSummaryItem(const ApplePayLineItem& lineItem)
{
    ASSERT(lineItem.paymentTiming == ApplePayPaymentTiming::Recurring);
    PKRecurringPaymentSummaryItem *summaryItem = [PAL::getPKRecurringPaymentSummaryItemClass() summaryItemWithLabel:lineItem.label amount:toDecimalNumber(lineItem.amount) type:toPKPaymentSummaryItemType(lineItem.type)];
    if (!lineItem.recurringPaymentStartDate.isNaN())
        summaryItem.startDate = toDate(lineItem.recurringPaymentStartDate);
    summaryItem.intervalUnit = toCalendarUnit(lineItem.recurringPaymentIntervalUnit);
    summaryItem.intervalCount = lineItem.recurringPaymentIntervalCount;
    if (!lineItem.recurringPaymentEndDate.isNaN())
        summaryItem.endDate = toDate(lineItem.recurringPaymentEndDate);
    return summaryItem;
}

#endif // HAVE(PASSKIT_RECURRING_SUMMARY_ITEM)

#if HAVE(PASSKIT_DEFERRED_SUMMARY_ITEM)

PKDeferredPaymentSummaryItem *platformDeferredSummaryItem(const ApplePayLineItem& lineItem)
{
    ASSERT(lineItem.paymentTiming == ApplePayPaymentTiming::Deferred);
    PKDeferredPaymentSummaryItem *summaryItem = [PAL::getPKDeferredPaymentSummaryItemClass() summaryItemWithLabel:lineItem.label amount:toDecimalNumber(lineItem.amount) type:toPKPaymentSummaryItemType(lineItem.type)];
    if (!lineItem.deferredPaymentDate.isNaN())
        summaryItem.deferredDate = toDate(lineItem.deferredPaymentDate);
    return summaryItem;
}

#endif // HAVE(PASSKIT_DEFERRED_SUMMARY_ITEM)

#if HAVE(PASSKIT_AUTOMATIC_RELOAD_SUMMARY_ITEM)

PKAutomaticReloadPaymentSummaryItem *platformAutomaticReloadSummaryItem(const ApplePayLineItem& lineItem)
{
    ASSERT(lineItem.paymentTiming == ApplePayPaymentTiming::AutomaticReload);
    PKAutomaticReloadPaymentSummaryItem *summaryItem = [PAL::getPKAutomaticReloadPaymentSummaryItemClass() summaryItemWithLabel:lineItem.label amount:toDecimalNumber(lineItem.amount) type:toPKPaymentSummaryItemType(lineItem.type)];
    summaryItem.thresholdAmount = toDecimalNumber(lineItem.automaticReloadPaymentThresholdAmount);
    return summaryItem;
}

#endif // HAVE(PASSKIT_AUTOMATIC_RELOAD_SUMMARY_ITEM)

#if HAVE(PASSKIT_DISBURSEMENTS)

PKDisbursementSummaryItem *platformDisbursementSummaryItem(const ApplePayLineItem& lineItem)
{
    ASSERT(lineItem.disbursementLineItemType == ApplePayLineItem::DisbursementLineItemType::Disbursement);
    PKDisbursementSummaryItem *summaryItem = [PAL::getPKDisbursementSummaryItemClass() summaryItemWithLabel:lineItem.label amount:toDecimalNumber(lineItem.amount)];
    return summaryItem;
}

PKInstantFundsOutFeeSummaryItem *platformInstantFundsOutFeeSummaryItem(const ApplePayLineItem& lineItem)
{
    ASSERT(lineItem.disbursementLineItemType == ApplePayLineItem::DisbursementLineItemType::InstantFundsOutFee);
    PKInstantFundsOutFeeSummaryItem *summaryItem = [PAL::getPKInstantFundsOutFeeSummaryItemClass() summaryItemWithLabel:lineItem.label amount:toDecimalNumber(lineItem.amount)];
    return summaryItem;
}

#endif // HAVE(PASSKIT_DISBURSEMENTS)

PKPaymentSummaryItem *platformSummaryItem(const ApplePayLineItem& lineItem)
{
#if HAVE(PASSKIT_DISBURSEMENTS)
    if (lineItem.disbursementLineItemType.has_value()) {
        switch (lineItem.disbursementLineItemType.value()) {
        case ApplePayLineItem::DisbursementLineItemType::Disbursement:
            return platformDisbursementSummaryItem(lineItem);
        case ApplePayLineItem::DisbursementLineItemType::InstantFundsOutFee:
            return platformInstantFundsOutFeeSummaryItem(lineItem);
        }
    }
#endif // HAVE(PASSKIT_DISBURSEMENTS)

    switch (lineItem.paymentTiming) {
    case ApplePayPaymentTiming::Immediate:
        break;

#if HAVE(PASSKIT_RECURRING_SUMMARY_ITEM)
    case ApplePayPaymentTiming::Recurring:
        return platformRecurringSummaryItem(lineItem);
#endif

#if HAVE(PASSKIT_DEFERRED_SUMMARY_ITEM)
    case ApplePayPaymentTiming::Deferred:
        return platformDeferredSummaryItem(lineItem);
#endif

#if HAVE(PASSKIT_AUTOMATIC_RELOAD_SUMMARY_ITEM)
    case ApplePayPaymentTiming::AutomaticReload:
        return platformAutomaticReloadSummaryItem(lineItem);
#endif
    }

    return [PAL::getPKPaymentSummaryItemClass() summaryItemWithLabel:lineItem.label amount:toDecimalNumber(lineItem.amount) type:toPKPaymentSummaryItemType(lineItem.type)];
}

#if HAVE(PASSKIT_DISBURSEMENTS)
// Disbursement Requests have a unique quirk: the total doesn't actually matter, we need to disregard any totals (this is a separate method to avoid confusion rather than making the total in `platformSummaryItems` optional
NSArray *platformDisbursementSummaryItems(const Vector<ApplePayLineItem>& lineItems)
{
    NSMutableArray *paymentSummaryItems = [NSMutableArray arrayWithCapacity:lineItems.size()];
    for (auto& lineItem : lineItems) {
        if (PKPaymentSummaryItem *summaryItem = platformSummaryItem(lineItem))
            [paymentSummaryItems addObject:summaryItem];
    }
    return adoptNS([paymentSummaryItems copy]).autorelease();
}
#endif // HAVE(PASSKIT_DISBURSEMENTS)

NSArray *platformSummaryItems(const ApplePayLineItem& total, const Vector<ApplePayLineItem>& lineItems)
{
    NSMutableArray *paymentSummaryItems = [NSMutableArray arrayWithCapacity:lineItems.size() + 1];
    for (auto& lineItem : lineItems) {
        if (PKPaymentSummaryItem *summaryItem = platformSummaryItem(lineItem))
            [paymentSummaryItems addObject:summaryItem];
    }

    if (PKPaymentSummaryItem *totalItem = platformSummaryItem(total))
        [paymentSummaryItems addObject:totalItem];

    return adoptNS([paymentSummaryItems copy]).autorelease();
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
