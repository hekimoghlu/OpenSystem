/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
#include "ApplePayShippingMethodSelectedEvent.h"

#if ENABLE(APPLE_PAY)

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ApplePayShippingMethodSelectedEvent);

ApplePayShippingMethodSelectedEvent::ApplePayShippingMethodSelectedEvent(const AtomString& type, const ApplePayShippingMethod& shippingMethod)
    : Event(EventInterfaceType::ApplePayShippingMethodSelectedEvent, type, CanBubble::No, IsCancelable::No)
    , m_shippingMethod(shippingMethod)
{
}

ApplePayShippingMethodSelectedEvent::~ApplePayShippingMethodSelectedEvent() = default;

}

#endif
