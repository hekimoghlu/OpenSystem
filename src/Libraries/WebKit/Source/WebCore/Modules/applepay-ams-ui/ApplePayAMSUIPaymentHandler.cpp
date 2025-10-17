/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#include "ApplePayAMSUIPaymentHandler.h"

#if ENABLE(APPLE_PAY_AMS_UI) && ENABLE(PAYMENT_REQUEST)

#include "ApplePayAMSUIRequest.h"
#include "Document.h"
#include "JSApplePayAMSUIRequest.h"
#include "JSDOMConvert.h"
#include "Page.h"

namespace WebCore {

static ExceptionOr<ApplePayAMSUIRequest> convertAndValidateApplePayAMSUIRequest(Document& document, JSC::JSValue data)
{
    if (data.isEmpty())
        return Exception { ExceptionCode::TypeError, "Missing payment method data."_s };

    auto throwScope = DECLARE_THROW_SCOPE(document.vm());
    auto applePayAMSUIRequestConversionResult = convertDictionary<ApplePayAMSUIRequest>(*document.globalObject(), data);
    if (applePayAMSUIRequestConversionResult.hasException(throwScope))
        return Exception { ExceptionCode::ExistingExceptionError };
    auto applePayAMSUIRequest = applePayAMSUIRequestConversionResult.releaseReturnValue();

    if (!applePayAMSUIRequest.engagementRequest.startsWith('{'))
        return Exception { ExceptionCode::TypeError, "Member ApplePayAMSUIRequest.engagementRequest is required and must be a JSON-serializable object"_s };

    return WTFMove(applePayAMSUIRequest);
}

ExceptionOr<void> ApplePayAMSUIPaymentHandler::validateData(Document& document, JSC::JSValue data)
{
    auto requestOrException = convertAndValidateApplePayAMSUIRequest(document, data);
    if (requestOrException.hasException())
        return requestOrException.releaseException();

    return { };
}

bool ApplePayAMSUIPaymentHandler::handlesIdentifier(const PaymentRequest::MethodIdentifier& identifier)
{
    if (!std::holds_alternative<URL>(identifier))
        return false;

    auto& url = std::get<URL>(identifier);
    return url.host() == "apple.com"_s && url.path() == "/ams-ui"_s;
}

bool ApplePayAMSUIPaymentHandler::hasActiveSession(Document& document)
{
    auto* page = document.page();
    return page && page->hasActiveApplePayAMSUISession();
}

void ApplePayAMSUIPaymentHandler::finishSession(std::optional<bool>&& result)
{
    if (!result) {
        m_paymentRequest->reject(Exception { ExceptionCode::AbortError });
        return;
    }

    m_paymentRequest->accept(std::get<URL>(m_identifier).string(), [success = *result] (JSC::JSGlobalObject& lexicalGlobalObject) -> JSC::Strong<JSC::JSObject> {
        JSC::JSLockHolder lock { &lexicalGlobalObject };

        JSC::VM& vm = lexicalGlobalObject.vm();
        auto throwScope = DECLARE_THROW_SCOPE(vm);

        auto* object = constructEmptyObject(&lexicalGlobalObject);
        object->putDirect(vm, JSC::Identifier::fromString(vm, "success"_s), JSC::jsBoolean(success));

        RETURN_IF_EXCEPTION(throwScope, { });

        return { vm, object };
    });
}

ApplePayAMSUIPaymentHandler::ApplePayAMSUIPaymentHandler(Document& document, const PaymentRequest::MethodIdentifier& identifier, PaymentRequest& paymentRequest)
    : ContextDestructionObserver { &document }
    , m_identifier { identifier }
    , m_paymentRequest { paymentRequest }
{
    ASSERT(handlesIdentifier(m_identifier));
}

Document& ApplePayAMSUIPaymentHandler::document() const
{
    ASSERT(scriptExecutionContext());
    return downcast<Document>(*scriptExecutionContext());
}

Page& ApplePayAMSUIPaymentHandler::page() const
{
    ASSERT(document().page());
    return *document().page();
}

ExceptionOr<void> ApplePayAMSUIPaymentHandler::convertData(Document& document, JSC::JSValue data)
{
    auto requestOrException = convertAndValidateApplePayAMSUIRequest(document, data);
    if (requestOrException.hasException())
        return requestOrException.releaseException();

    m_applePayAMSUIRequest = requestOrException.releaseReturnValue();
    return { };
}

ExceptionOr<void> ApplePayAMSUIPaymentHandler::show(Document&)
{
    ASSERT(m_applePayAMSUIRequest);

    if (!page().startApplePayAMSUISession(page().mainFrameURL(), *this, *m_applePayAMSUIRequest))
        return Exception { ExceptionCode::AbortError };

    return { };
}

void ApplePayAMSUIPaymentHandler::hide()
{
    page().abortApplePayAMSUISession(*this);
}

void ApplePayAMSUIPaymentHandler::canMakePayment(Document& document, Function<void(bool)>&& completionHandler)
{
    auto* page = document.page();
    completionHandler(page && !page->hasActiveApplePayAMSUISession());
}

ExceptionOr<void> ApplePayAMSUIPaymentHandler::detailsUpdated(PaymentRequest::UpdateReason, String&& /* error */, AddressErrors&&, PayerErrorFields&&, JSC::JSObject* /* paymentMethodErrors */)
{
    ASSERT_NOT_REACHED_WITH_MESSAGE("ApplePayAMSUIPaymentHandler does not need shipping/payment info");
    return { };
}

ExceptionOr<void> ApplePayAMSUIPaymentHandler::merchantValidationCompleted(JSC::JSValue&& /* merchantSessionValue */)
{
    ASSERT_NOT_REACHED_WITH_MESSAGE("ApplePayAMSUIPaymentHandler does not need merchant validation");
    return { };
}

ExceptionOr<void> ApplePayAMSUIPaymentHandler::complete(Document&, std::optional<PaymentComplete>&&, String&&)
{
    hide();
    return { };
}

ExceptionOr<void> ApplePayAMSUIPaymentHandler::retry(PaymentValidationErrors&&)
{
    return show(document());
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY) && ENABLE(PAYMENT_REQUEST)
