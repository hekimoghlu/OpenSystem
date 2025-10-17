/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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

#if HAVE(PASSKIT_DISBURSEMENTS)

#include <wtf/Forward.h>

OBJC_CLASS PKDisbursementPaymentRequest;

namespace WebCore {
class ApplePaySessionPaymentRequest;
enum class ApplePayContactField : uint8_t;
}

namespace WebKit {

RetainPtr<PKDisbursementPaymentRequest> platformDisbursementRequest(const WebCore::ApplePaySessionPaymentRequest&, const URL& originatingURL, const std::optional<Vector<WebCore::ApplePayContactField>>& requiredrecipientContactFields);

} // namespace WebKit

#endif // HAVE(PASSKIT_DISBURSEMENTS)
