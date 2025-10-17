/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#include "MerchantValidationEvent.h"

#if ENABLE(PAYMENT_REQUEST)

#include "Document.h"
#include "PaymentRequest.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MerchantValidationEvent);

Ref<MerchantValidationEvent> MerchantValidationEvent::create(const AtomString& type, const String& methodName, URL&& validationURL)
{
    return adoptRef(*new MerchantValidationEvent(type, methodName, WTFMove(validationURL)));
}

ExceptionOr<Ref<MerchantValidationEvent>> MerchantValidationEvent::create(Document& document, const AtomString& type, Init&& eventInit)
{
    auto validationURL = document.completeURL(eventInit.validationURL, ScriptExecutionContext::ForceUTF8::Yes);
    if (!validationURL.isValid())
        return Exception { ExceptionCode::TypeError };

    auto methodName = WTFMove(eventInit.methodName);
    if (!methodName.isEmpty()) {
        auto validatedMethodName = convertAndValidatePaymentMethodIdentifier(methodName);
        if (!validatedMethodName)
            return Exception { ExceptionCode::RangeError, makeString('"', methodName, "\" is an invalid payment method identifier."_s) };
    }

    return adoptRef(*new MerchantValidationEvent(type, WTFMove(methodName), WTFMove(validationURL), WTFMove(eventInit)));
}

MerchantValidationEvent::MerchantValidationEvent(const AtomString& type, const String& methodName, URL&& validationURL)
    : Event { EventInterfaceType::MerchantValidationEvent, type, Event::CanBubble::No, Event::IsCancelable::No }
    , m_methodName { methodName }
    , m_validationURL { WTFMove(validationURL) }
{
    ASSERT(isTrusted());
    ASSERT(m_validationURL.isValid());
}

MerchantValidationEvent::MerchantValidationEvent(const AtomString& type, String&& methodName, URL&& validationURL, Init&& eventInit)
    : Event { EventInterfaceType::MerchantValidationEvent, type, WTFMove(eventInit), IsTrusted::No }
    , m_methodName { WTFMove(methodName) }
    , m_validationURL { WTFMove(validationURL) }
{
    ASSERT(!isTrusted());
    ASSERT(m_validationURL.isValid());
}

ExceptionOr<void> MerchantValidationEvent::complete(Ref<DOMPromise>&& merchantSessionPromise)
{
    if (!isTrusted())
        return Exception { ExceptionCode::InvalidStateError };

    if (m_isCompleted)
        return Exception { ExceptionCode::InvalidStateError };

    auto exception = downcast<PaymentRequest>(target())->completeMerchantValidation(*this, WTFMove(merchantSessionPromise));
    if (exception.hasException())
        return exception.releaseException();

    m_isCompleted = true;
    return { };
}

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
