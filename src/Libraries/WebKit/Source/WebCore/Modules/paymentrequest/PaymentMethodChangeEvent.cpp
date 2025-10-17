/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#include "PaymentMethodChangeEvent.h"

#if ENABLE(PAYMENT_REQUEST)

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PaymentMethodChangeEvent);

PaymentMethodChangeEvent::PaymentMethodChangeEvent(const AtomString& type, Init&& eventInit)
    : PaymentRequestUpdateEvent { EventInterfaceType::PaymentMethodChangeEvent, type, eventInit }
    , m_methodName { WTFMove(eventInit.methodName) }
    , m_methodDetails { std::in_place_type_t<JSValueInWrappedObject>(), eventInit.methodDetails.get() }
{
}

PaymentMethodChangeEvent::PaymentMethodChangeEvent(const AtomString& type, const String& methodName, MethodDetailsFunction&& methodDetailsFunction)
    : PaymentRequestUpdateEvent { EventInterfaceType::PaymentMethodChangeEvent, type }
    , m_methodName { methodName }
    , m_methodDetails { std::in_place_type_t<MethodDetailsFunction>(), WTFMove(methodDetailsFunction) }
{
}

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
