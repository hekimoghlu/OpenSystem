/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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

#if ENABLE(APPLE_PAY) && ENABLE(PAYMENT_REQUEST)

#include "ApplePayPaymentMethodType.h"
#include "ApplePayRequest.h"
#include "ContextDestructionObserver.h"
#include "PaymentHandler.h"
#include "PaymentRequest.h"
#include "PaymentSession.h"
#include <wtf/Function.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class ApplePayError;
class PaymentCoordinator;
struct ApplePayModifier;
struct PaymentDetailsModifier;

class ApplePayPaymentHandler final : public PaymentHandler, public PaymentSession, private ContextDestructionObserver {
public:
    static ExceptionOr<void> validateData(Document&, JSC::JSValue);
    static bool handlesIdentifier(const PaymentRequest::MethodIdentifier&);
    static bool hasActiveSession(Document&);

private:
    friend class PaymentHandler;
    explicit ApplePayPaymentHandler(Document&, const PaymentRequest::MethodIdentifier&, PaymentRequest&);

    Document& document() const;
    PaymentCoordinator& paymentCoordinator() const;

    ExceptionOr<Vector<ApplePayShippingMethod>> computeShippingMethods() const;
    ExceptionOr<std::tuple<ApplePayLineItem, Vector<ApplePayLineItem>>> computeTotalAndLineItems() const;
    Vector<Ref<ApplePayError>> computeErrors(String&& error, AddressErrors&&, PayerErrorFields&&, JSC::JSObject* paymentMethodErrors) const;
    Vector<Ref<ApplePayError>> computeErrors(JSC::JSObject* paymentMethodErrors) const;
    void computeAddressErrors(String&& error, AddressErrors&&, Vector<Ref<ApplePayError>>&) const;
    void computePayerErrors(PayerErrorFields&&, Vector<Ref<ApplePayError>>&) const;
    ExceptionOr<void> computePaymentMethodErrors(JSC::JSObject* paymentMethodErrors, Vector<Ref<ApplePayError>>&) const;
    ExceptionOr<std::optional<std::tuple<PaymentDetailsModifier, ApplePayModifier>>> firstApplicableModifier() const;

    ExceptionOr<void> shippingAddressUpdated(Vector<Ref<ApplePayError>>&& errors);
    ExceptionOr<void> shippingOptionUpdated();
    ExceptionOr<void> paymentMethodUpdated(Vector<Ref<ApplePayError>>&& errors);

    // PaymentHandler
    ExceptionOr<void> convertData(Document&, JSC::JSValue) final;
    ExceptionOr<void> show(Document&) final;
    bool canAbortSession() final { return true; }
    void hide() final;
    void canMakePayment(Document&, Function<void(bool)>&& completionHandler) final;
    ExceptionOr<void> detailsUpdated(PaymentRequest::UpdateReason, String&& error, AddressErrors&&, PayerErrorFields&&, JSC::JSObject* paymentMethodErrors) final;
    ExceptionOr<void> merchantValidationCompleted(JSC::JSValue&&) final;
    ExceptionOr<void> complete(Document&, std::optional<PaymentComplete>&&, String&& serializedData) final;
    ExceptionOr<void> retry(PaymentValidationErrors&&) final;

    // PaymentSession
    unsigned version() const final;
    void validateMerchant(URL&&) final;
    void didAuthorizePayment(const Payment&) final;
    void didSelectShippingMethod(const ApplePayShippingMethod&) final;
    void didSelectShippingContact(const PaymentContact&) final;
    void didSelectPaymentMethod(const PaymentMethod&) final;
#if ENABLE(APPLE_PAY_COUPON_CODE)
    void didChangeCouponCode(String&& couponCode) final;
#endif
    void didCancelPaymentSession(PaymentSessionError&&) final;

    PaymentRequest::MethodIdentifier m_identifier;
    Ref<PaymentRequest> m_paymentRequest;
    std::optional<ApplePayRequest> m_applePayRequest;
    std::optional<ApplePayPaymentMethodType> m_selectedPaymentMethodType;

    enum class UpdateState : uint8_t {
        None,
        ShippingAddress,
        ShippingOption,
        PaymentMethod,
#if ENABLE(APPLE_PAY_COUPON_CODE)
        CouponCode,
#endif
    } m_updateState { UpdateState::None };
};

} // namespace WebCore

#endif // ENABLE(APPLE_PAY) && ENABLE(PAYMENT_REQUEST)
