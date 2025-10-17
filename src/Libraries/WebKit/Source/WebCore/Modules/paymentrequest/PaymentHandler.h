/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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

#if ENABLE(PAYMENT_REQUEST)

#include "PaymentRequest.h"
#include "PaymentSessionBase.h"
#include <wtf/Function.h>

namespace JSC {
class JSValue;
}

namespace WebCore {

class Document;
struct AddressErrors;
struct PayerErrorFields;
struct PaymentValidationErrors;

class PaymentHandler : public virtual PaymentSessionBase {
public:
    static RefPtr<PaymentHandler> create(Document&, PaymentRequest&, const PaymentRequest::MethodIdentifier&);
    static ExceptionOr<void> canCreateSession(Document&);
    static ExceptionOr<void> validateData(Document&, JSC::JSValue, const PaymentRequest::MethodIdentifier&);
    static bool enabledForContext(ScriptExecutionContext&);
    static bool hasActiveSession(Document&);

    virtual ExceptionOr<void> convertData(Document&, JSC::JSValue) = 0;
    virtual ExceptionOr<void> show(Document&) = 0;
    virtual bool canAbortSession() = 0;
    virtual void hide() = 0;
    virtual void canMakePayment(Document&, Function<void(bool)>&& completionHandler) = 0;
    virtual ExceptionOr<void> detailsUpdated(PaymentRequest::UpdateReason, String&& error, AddressErrors&&, PayerErrorFields&&, JSC::JSObject* paymentMethodErrors) = 0;
    virtual ExceptionOr<void> merchantValidationCompleted(JSC::JSValue&&) = 0;
    virtual ExceptionOr<void> complete(Document&, std::optional<PaymentComplete>&&, String&& serializedData) = 0;
    virtual ExceptionOr<void> retry(PaymentValidationErrors&&) = 0;
};

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
