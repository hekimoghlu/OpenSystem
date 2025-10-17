/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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

#include "Event.h"
#include "PaymentSessionError.h"

namespace WebCore {

class PaymentSessionError;

class ApplePayCancelEvent : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ApplePayCancelEvent);
public:
    static Ref<ApplePayCancelEvent> create(const AtomString& type, PaymentSessionError&& sessionError)
    {
        return adoptRef(*new ApplePayCancelEvent(type, WTFMove(sessionError)));
    }

    ApplePaySessionError sessionError() const;

private:
    explicit ApplePayCancelEvent(const AtomString&, PaymentSessionError&&);

    PaymentSessionError m_sessionError;
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
