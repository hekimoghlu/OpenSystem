/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#include "config.h"
#include "ApplePayMerchantCapability.h"

#if ENABLE(APPLE_PAY)

namespace WebCore {

ExceptionOr<ApplePaySessionPaymentRequest::MerchantCapabilities> convertAndValidate(const Vector<ApplePayMerchantCapability>& merchantCapabilities)
{
    if (merchantCapabilities.isEmpty())
        return Exception { ExceptionCode::TypeError, "At least one merchant capability must be provided."_s };

    ApplePaySessionPaymentRequest::MerchantCapabilities result;

    for (auto& merchantCapability : merchantCapabilities) {
        switch (merchantCapability) {
        case ApplePayMerchantCapability::Supports3DS:
            result.supports3DS = true;
            break;
        case ApplePayMerchantCapability::SupportsEMV:
            result.supportsEMV = true;
            break;
        case ApplePayMerchantCapability::SupportsCredit:
            result.supportsCredit = true;
            break;
        case ApplePayMerchantCapability::SupportsDebit:
            result.supportsDebit = true;
            break;
#if ENABLE(APPLE_PAY_DISBURSEMENTS)
        case ApplePayMerchantCapability::SupportsInstantFundsOut:
            result.supportsInstantFundsOut = true;
            break;
#endif
        }
    }

    return WTFMove(result);
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
