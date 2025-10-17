/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#import "PaymentAPIVersion.h"

#if ENABLE(APPLE_PAY)

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebCore {

unsigned PaymentAPIVersion::current()
{
    static unsigned current = [] {
#if ENABLE(APPLE_PAY_FEATURES)
        // This version number should not be changed anymore, features can now be found in method.data.features.
        return 15;
#elif ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_LINE_ITEM) || ENABLE(APPLE_PAY_RECURRING_PAYMENTS) || ENABLE(APPLE_PAY_AUTOMATIC_RELOAD_PAYMENTS) || ENABLE(APPLE_PAY_MULTI_MERCHANT_PAYMENTS) || ENABLE(APPLE_PAY_PAYMENT_ORDER_DETAILS)
        return 14;
#elif ENABLE(APPLE_PAY_SELECTED_SHIPPING_METHOD) || ENABLE(APPLE_PAY_AMS_UI)
        return 13;
#elif ENABLE(APPLE_PAY_COUPON_CODE) || ENABLE(APPLE_PAY_SHIPPING_CONTACT_EDITING_MODE) || ENABLE(APPLE_PAY_RECURRING_LINE_ITEM) || ENABLE(APPLE_PAY_DEFERRED_LINE_ITEM) || ENABLE(APPLE_PAY_SHIPPING_METHOD_DATE_COMPONENTS_RANGE)
        return 12;
#elif ENABLE(APPLE_PAY_SESSION_V11)
        return 11;
#elif HAVE(PASSKIT_NEW_BUTTON_TYPES)
        return 10;
#elif HAVE(PASSKIT_INSTALLMENTS)
        if (PAL::getPKPaymentInstallmentConfigurationClass()) {
            if (PAL::getPKPaymentInstallmentItemClass())
                return 9;
            return 8;
        }
#endif
        return 7;
    }();
    return current;
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
