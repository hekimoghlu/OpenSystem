/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "IDLTypes.h"
#include "PaymentDetailsInit.h"
#include "PaymentMethodChangeEvent.h"
#include "PaymentOptions.h"
#include "PaymentResponse.h"
#include <variant>
#include <wtf/URL.h>

namespace WebCore {

class Document;
class Event;
class PaymentAddress;
class PaymentHandler;
class PaymentRequestUpdateEvent;
class PaymentResponse;
enum class PaymentComplete;
enum class PaymentShippingType;
struct PaymentDetailsUpdate;
struct PaymentMethodData;
template<typename IDLType> class DOMPromiseDeferred;

class PaymentRequest final : public ActiveDOMObject, public EventTarget, public RefCounted<PaymentRequest> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PaymentRequest);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    using AbortPromise = DOMPromiseDeferred<void>;
    using CanMakePaymentPromise = DOMPromiseDeferred<IDLBoolean>;
    using ShowPromise = DOMPromiseDeferred<IDLInterface<PaymentResponse>>;

    static ExceptionOr<Ref<PaymentRequest>> create(Document&, Vector<PaymentMethodData>&&, PaymentDetailsInit&&, PaymentOptions&&);
    ~PaymentRequest();

    void show(Document&, RefPtr<DOMPromise>&& detailsPromise, ShowPromise&&);
    void abort(AbortPromise&&);
    void canMakePayment(Document&, CanMakePaymentPromise&&);

    const String& id() const;
    PaymentAddress* shippingAddress() const { return m_shippingAddress.get(); }
    const String& shippingOption() const { return m_shippingOption; }
    std::optional<PaymentShippingType> shippingType() const;

    enum class State {
        Created,
        Interactive,
        Closed,
    };

    enum class UpdateReason {
        ShowDetailsResolved,
        ShippingAddressChanged,
        ShippingOptionChanged,
        PaymentMethodChanged,
    };

    State state() const { return m_state; }

    const PaymentOptions& paymentOptions() const { return m_options; }
    const PaymentDetailsInit& paymentDetails() const { return m_details; }
    const Vector<String>& serializedModifierData() const { return m_serializedModifierData; }

    void shippingAddressChanged(Ref<PaymentAddress>&&);
    void shippingOptionChanged(const String& shippingOption);
    void paymentMethodChanged(const String& methodName, PaymentMethodChangeEvent::MethodDetailsFunction&&);
    ExceptionOr<void> updateWith(UpdateReason, Ref<DOMPromise>&&);
    ExceptionOr<void> completeMerchantValidation(Event&, Ref<DOMPromise>&&);
    void accept(const String& methodName, PaymentResponse::DetailsFunction&&);
    void accept(const String& methodName, PaymentResponse::DetailsFunction&&, Ref<PaymentAddress>&& shippingAddress, const String& payerName, const String& payerEmail, const String& payerPhone);
    void reject(Exception&&);
    ExceptionOr<void> complete(Document&, std::optional<PaymentComplete>&&, String&& serializedData);
    ExceptionOr<void> retry(PaymentValidationErrors&&);
    void cancel();

    using MethodIdentifier = std::variant<String, URL>;

private:
    struct Method {
        MethodIdentifier identifier;
        String serializedData;
    };

    struct PaymentHandlerWithPendingActivity {
        Ref<PaymentHandler> paymentHandler;
        Ref<PendingActivity<PaymentRequest>> pendingActivity;
    };

    PaymentRequest(Document&, PaymentOptions&&, PaymentDetailsInit&&, Vector<String>&& serializedModifierData, Vector<Method>&& serializedMethodData, String&& selectedShippingOption);

    void dispatchAndCheckUpdateEvent(Ref<PaymentRequestUpdateEvent>&&);
    void settleDetailsPromise(UpdateReason);
    void whenDetailsSettled(std::function<void()>&&);
    void abortWithException(Exception&&);
    PaymentHandler* activePaymentHandler() { return m_activePaymentHandler ? m_activePaymentHandler->paymentHandler.ptr() : nullptr; }
    void settleShowPromise(ExceptionOr<PaymentResponse&>&&);
    void closeActivePaymentHandler();

    // ActiveDOMObject
    void stop() final;
    void suspend(ReasonForSuspension) final;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::PaymentRequest; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    bool isPaymentRequest() const final { return true; }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    PaymentOptions m_options;
    PaymentDetailsInit m_details;
    Vector<String> m_serializedModifierData;
    Vector<Method> m_serializedMethodData;
    String m_shippingOption;
    RefPtr<PaymentAddress> m_shippingAddress;
    State m_state { State::Created };
    std::unique_ptr<ShowPromise> m_showPromise;
    std::optional<PaymentHandlerWithPendingActivity> m_activePaymentHandler;
    RefPtr<DOMPromise> m_detailsPromise;
    RefPtr<DOMPromise> m_merchantSessionPromise;
    RefPtr<PaymentResponse> m_response;
    bool m_isUpdating { false };
    bool m_isCancelPending { false };
};

std::optional<PaymentRequest::MethodIdentifier> convertAndValidatePaymentMethodIdentifier(const String& identifier);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PaymentRequest)
    static bool isType(const WebCore::EventTarget& eventTarget) { return eventTarget.isPaymentRequest(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(PAYMENT_REQUEST)
