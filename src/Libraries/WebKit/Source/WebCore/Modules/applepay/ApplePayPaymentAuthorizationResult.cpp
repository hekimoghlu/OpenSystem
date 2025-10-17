/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#include "ApplePayPaymentAuthorizationResult.h"

#if ENABLE(APPLE_PAY)

namespace WebCore {

bool ApplePayPaymentAuthorizationResult::isFinalState() const
{
    switch (status) {
    case ApplePayPaymentAuthorizationResult::Success:
    case ApplePayPaymentAuthorizationResult::Failure:
        return errors.isEmpty();

    case ApplePayPaymentAuthorizationResult::PINRequired:
    case ApplePayPaymentAuthorizationResult::PINIncorrect:
    case ApplePayPaymentAuthorizationResult::PINLockout:
        return false;

    default:
        ASSERT_NOT_REACHED();
        return false;
    }
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
