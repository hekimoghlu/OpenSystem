/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include "WebPaymentCoordinator.h"

#if ENABLE(APPLE_PAY)

#include "ApplePayPaymentSetupFeaturesWebKit.h"
#include "MessageSenderInlines.h"
#include "PaymentSetupConfigurationWebKit.h"
#include "WebPage.h"
#include "WebPaymentCoordinatorMessages.h"
#include "WebPaymentCoordinatorProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/ApplePayCouponCodeUpdate.h>
#include <WebCore/ApplePayPaymentAuthorizationResult.h>
#include <WebCore/ApplePayPaymentMethodUpdate.h>
#include <WebCore/ApplePayShippingContactUpdate.h>
#include <WebCore/ApplePayShippingMethodUpdate.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/PaymentCoordinator.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPaymentCoordinator);

Ref<WebPaymentCoordinator> WebPaymentCoordinator::create(WebPage& webPage)
{
    return adoptRef(*new WebPaymentCoordinator(webPage));
}

WebPaymentCoordinator::WebPaymentCoordinator(WebPage& webPage)
    : m_webPage(webPage)
{
    WebProcess::singleton().addMessageReceiver(Messages::WebPaymentCoordinator::messageReceiverName(), webPage.identifier(), *this);
}

WebPaymentCoordinator::~WebPaymentCoordinator()
{
    WebProcess::singleton().removeMessageReceiver(*this);
}

void WebPaymentCoordinator::networkProcessConnectionClosed()
{
#if ENABLE(APPLE_PAY_REMOTE_UI)
    didCancelPaymentSession({ });
#endif
}

std::optional<String> WebPaymentCoordinator::validatedPaymentNetwork(const String& paymentNetwork) const
{
    if (!m_availablePaymentNetworks)
        m_availablePaymentNetworks = platformAvailablePaymentNetworks();

    auto result = m_availablePaymentNetworks->find(paymentNetwork);
    if (result == m_availablePaymentNetworks->end())
        return std::nullopt;
    return *result;
}

bool WebPaymentCoordinator::canMakePayments()
{
    auto now = MonotonicTime::now();
    if (now - m_timestampOfLastCanMakePaymentsRequest > 1_min || !m_lastCanMakePaymentsResult) {
        auto sendResult = sendSync(Messages::WebPaymentCoordinatorProxy::CanMakePayments());
        if (!sendResult.succeeded())
            return false;
        auto [canMakePayments] = sendResult.takeReply();

        m_timestampOfLastCanMakePaymentsRequest = now;
        m_lastCanMakePaymentsResult = canMakePayments;
    }
    return *m_lastCanMakePaymentsResult;
}

void WebPaymentCoordinator::canMakePaymentsWithActiveCard(const String& merchantIdentifier, const String& domainName, CompletionHandler<void(bool)>&& completionHandler)
{
    sendWithAsyncReply(Messages::WebPaymentCoordinatorProxy::CanMakePaymentsWithActiveCard(merchantIdentifier, domainName), WTFMove(completionHandler));
}

void WebPaymentCoordinator::openPaymentSetup(const String& merchantIdentifier, const String& domainName, CompletionHandler<void(bool)>&& completionHandler)
{
    sendWithAsyncReply(Messages::WebPaymentCoordinatorProxy::OpenPaymentSetup(merchantIdentifier, domainName), WTFMove(completionHandler));
}

bool WebPaymentCoordinator::showPaymentUI(const URL& originatingURL, const Vector<URL>& linkIconURLs, const WebCore::ApplePaySessionPaymentRequest& paymentRequest)
{
    RefPtr webPage = m_webPage.get();
    if (!webPage)
        return false;

    auto linkIconURLStrings = linkIconURLs.map([](auto& linkIconURL) {
        return linkIconURL.string();
    });

    auto sendResult = sendSync(Messages::WebPaymentCoordinatorProxy::ShowPaymentUI(webPage->identifier(), webPage->webPageProxyIdentifier(), originatingURL.string(), linkIconURLStrings, paymentRequest));
    auto [result] = sendResult.takeReplyOr(false);
    return result;
}

void WebPaymentCoordinator::completeMerchantValidation(const WebCore::PaymentMerchantSession& paymentMerchantSession)
{
    send(Messages::WebPaymentCoordinatorProxy::CompleteMerchantValidation(paymentMerchantSession));
}

void WebPaymentCoordinator::completeShippingMethodSelection(std::optional<WebCore::ApplePayShippingMethodUpdate>&& update)
{
    send(Messages::WebPaymentCoordinatorProxy::CompleteShippingMethodSelection(WTFMove(update)));
}

void WebPaymentCoordinator::completeShippingContactSelection(std::optional<WebCore::ApplePayShippingContactUpdate>&& update)
{
    send(Messages::WebPaymentCoordinatorProxy::CompleteShippingContactSelection(WTFMove(update)));
}

void WebPaymentCoordinator::completePaymentMethodSelection(std::optional<WebCore::ApplePayPaymentMethodUpdate>&& update)
{
    send(Messages::WebPaymentCoordinatorProxy::CompletePaymentMethodSelection(WTFMove(update)));
}

#if ENABLE(APPLE_PAY_COUPON_CODE)

void WebPaymentCoordinator::completeCouponCodeChange(std::optional<WebCore::ApplePayCouponCodeUpdate>&& update)
{
    send(Messages::WebPaymentCoordinatorProxy::CompleteCouponCodeChange(WTFMove(update)));
}

#endif // ENABLE(APPLE_PAY_COUPON_CODE)

void WebPaymentCoordinator::completePaymentSession(WebCore::ApplePayPaymentAuthorizationResult&& result)
{
    send(Messages::WebPaymentCoordinatorProxy::CompletePaymentSession(WTFMove(result)));
}

void WebPaymentCoordinator::abortPaymentSession()
{
    send(Messages::WebPaymentCoordinatorProxy::AbortPaymentSession());
}

void WebPaymentCoordinator::cancelPaymentSession()
{
    send(Messages::WebPaymentCoordinatorProxy::CancelPaymentSession());
}

IPC::Connection* WebPaymentCoordinator::messageSenderConnection() const
{
#if ENABLE(APPLE_PAY_REMOTE_UI)
    return &WebProcess::singleton().ensureNetworkProcessConnection().connection();
#else
    return WebProcess::singleton().parentProcessConnection();
#endif
}

uint64_t WebPaymentCoordinator::messageSenderDestinationID() const
{
    return m_webPage ? m_webPage->identifier().toUInt64() : 0;
}

void WebPaymentCoordinator::validateMerchant(const String& validationURLString)
{
    paymentCoordinator().validateMerchant(URL { validationURLString });
}

void WebPaymentCoordinator::didAuthorizePayment(const WebCore::Payment& payment)
{
    paymentCoordinator().didAuthorizePayment(payment);
}

void WebPaymentCoordinator::didSelectShippingMethod(const WebCore::ApplePayShippingMethod& shippingMethod)
{
    paymentCoordinator().didSelectShippingMethod(shippingMethod);
}

void WebPaymentCoordinator::didSelectShippingContact(const WebCore::PaymentContact& shippingContact)
{
    paymentCoordinator().didSelectShippingContact(shippingContact);
}

void WebPaymentCoordinator::didSelectPaymentMethod(const WebCore::PaymentMethod& paymentMethod)
{
    paymentCoordinator().didSelectPaymentMethod(paymentMethod);
}

#if ENABLE(APPLE_PAY_COUPON_CODE)

void WebPaymentCoordinator::didChangeCouponCode(String&& couponCode)
{
    paymentCoordinator().didChangeCouponCode(WTFMove(couponCode));
}

#endif // ENABLE(APPLE_PAY_COUPON_CODE)

void WebPaymentCoordinator::didCancelPaymentSession(WebCore::PaymentSessionError&& sessionError)
{
    paymentCoordinator().didCancelPaymentSession(WTFMove(sessionError));
}

WebCore::PaymentCoordinator& WebPaymentCoordinator::paymentCoordinator()
{
    return m_webPage->corePage()->paymentCoordinator();
}

void WebPaymentCoordinator::getSetupFeatures(const WebCore::ApplePaySetupConfiguration& configuration, const URL& url, CompletionHandler<void(Vector<Ref<WebCore::ApplePaySetupFeature>>&&)>&& completionHandler)
{
    RefPtr webPage = m_webPage.get();
    if (!webPage)
        return completionHandler({ });
    webPage->sendWithAsyncReply(Messages::WebPaymentCoordinatorProxy::GetSetupFeatures(PaymentSetupConfiguration { configuration, url }), WTFMove(completionHandler));
}

void WebPaymentCoordinator::beginApplePaySetup(const WebCore::ApplePaySetupConfiguration& configuration, const URL& url, Vector<Ref<WebCore::ApplePaySetupFeature>>&& features, CompletionHandler<void(bool)>&& completionHandler)
{
    RefPtr webPage = m_webPage.get();
    if (!webPage)
        return completionHandler(false);
    webPage->sendWithAsyncReply(Messages::WebPaymentCoordinatorProxy::BeginApplePaySetup(PaymentSetupConfiguration { configuration, url }, PaymentSetupFeatures { WTFMove(features) }), WTFMove(completionHandler));
}

void WebPaymentCoordinator::endApplePaySetup()
{
    if (RefPtr webPage = m_webPage.get())
        webPage->send(Messages::WebPaymentCoordinatorProxy::EndApplePaySetup());
}

}

#endif
