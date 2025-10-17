/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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

#if ENABLE(APPLE_PAY_AMS_UI) && ENABLE(PAYMENT_REQUEST)

#include "ApplePayAMSUIRequest.h"
#include "ContextDestructionObserver.h"
#include "ExceptionOr.h"
#include "PaymentHandler.h"
#include "PaymentRequest.h"
#include <wtf/Function.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSObject;
class JSValue;
}

namespace WebCore {

class Document;
class Page;

struct AddressErrors;
struct PayerErrorFields;

enum class PaymentComplete;

class ApplePayAMSUIPaymentHandler final : public PaymentHandler, private ContextDestructionObserver {
public:
    static ExceptionOr<void> validateData(Document&, JSC::JSValue);
    static bool handlesIdentifier(const PaymentRequest::MethodIdentifier&);
    static bool hasActiveSession(Document&);

    void finishSession(std::optional<bool>&&);

private:
    friend class PaymentHandler;
    explicit ApplePayAMSUIPaymentHandler(Document&, const PaymentRequest::MethodIdentifier&, PaymentRequest&);

    Document& document() const;
    Page& page() const;

    // PaymentHandler
    ExceptionOr<void> convertData(Document&, JSC::JSValue) final;
    ExceptionOr<void> show(Document&) final;
    bool canAbortSession() final { return false; }
    void hide() final;
    void canMakePayment(Document&, Function<void(bool)>&& completionHandler) final;
    ExceptionOr<void> detailsUpdated(PaymentRequest::UpdateReason, String&& error, AddressErrors&&, PayerErrorFields&&, JSC::JSObject* paymentMethodErrors) final;
    ExceptionOr<void> merchantValidationCompleted(JSC::JSValue&&) final;
    ExceptionOr<void> complete(Document&, std::optional<PaymentComplete>&&, String&& serializedData) final;
    ExceptionOr<void> retry(PaymentValidationErrors&&) final;

    PaymentRequest::MethodIdentifier m_identifier;
    Ref<PaymentRequest> m_paymentRequest;
    std::optional<ApplePayAMSUIRequest> m_applePayAMSUIRequest;
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY) && ENABLE(PAYMENT_REQUEST)
