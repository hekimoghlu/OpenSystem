/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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

#include "ApplePayDateComponentsRange.h"
#include <optional>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct ApplePayShippingMethod final {
    String label;
    String detail;
    String amount;
    String identifier;

#if ENABLE(APPLE_PAY_SHIPPING_METHOD_DATE_COMPONENTS_RANGE)
    std::optional<ApplePayDateComponentsRange> dateComponentsRange;
#endif

#if ENABLE(APPLE_PAY_SELECTED_SHIPPING_METHOD)
    bool selected { false };
#endif
};

} // namespace WebCore

#endif
