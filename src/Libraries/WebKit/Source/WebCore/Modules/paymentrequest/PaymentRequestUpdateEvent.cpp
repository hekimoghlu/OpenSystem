/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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
#include "PaymentRequestUpdateEvent.h"

#if ENABLE(PAYMENT_REQUEST)

#include "EventNames.h"
#include "PaymentRequest.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PaymentRequestUpdateEvent);

PaymentRequestUpdateEvent::PaymentRequestUpdateEvent(enum EventInterfaceType eventInterface, const AtomString& type, const PaymentRequestUpdateEventInit& eventInit)
    : Event { eventInterface, type, eventInit, IsTrusted::No }
{
    ASSERT(!isTrusted());
}

PaymentRequestUpdateEvent::PaymentRequestUpdateEvent(enum EventInterfaceType eventInterface, const AtomString& type)
    : Event { eventInterface, type, CanBubble::No, IsCancelable::No }
{
    ASSERT(isTrusted());
}

PaymentRequestUpdateEvent::~PaymentRequestUpdateEvent() = default;

ExceptionOr<void> PaymentRequestUpdateEvent::updateWith(Ref<DOMPromise>&& detailsPromise)
{
    if (!isTrusted())
        return Exception { ExceptionCode::InvalidStateError };

    if (m_waitForUpdate)
        return Exception { ExceptionCode::InvalidStateError };

    stopPropagation();
    stopImmediatePropagation();

    PaymentRequest::UpdateReason reason;
    if (type() == eventNames().shippingaddresschangeEvent)
        reason = PaymentRequest::UpdateReason::ShippingAddressChanged;
    else if (type() == eventNames().shippingoptionchangeEvent)
        reason = PaymentRequest::UpdateReason::ShippingOptionChanged;
    else if (type() == eventNames().paymentmethodchangeEvent)
        reason = PaymentRequest::UpdateReason::PaymentMethodChanged;
    else {
        ASSERT_NOT_REACHED();
        return Exception { ExceptionCode::TypeError };
    }

    auto exception = downcast<PaymentRequest>(target())->updateWith(reason, WTFMove(detailsPromise));
    if (exception.hasException())
        return exception.releaseException();

    m_waitForUpdate = true;
    return { };
}

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
