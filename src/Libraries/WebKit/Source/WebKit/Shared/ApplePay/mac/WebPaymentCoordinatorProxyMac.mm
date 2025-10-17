/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#import "config.h"
#import "WebPaymentCoordinatorProxy.h"

#if PLATFORM(MAC) && ENABLE(APPLE_PAY)

#import "PaymentAuthorizationViewController.h"
#import "WebPageProxy.h"
#import <pal/cocoa/PassKitSoftLink.h>
#import <wtf/BlockPtr.h>

namespace WebKit {

void WebPaymentCoordinatorProxy::platformCanMakePayments(CompletionHandler<void(bool)>&& completionHandler)
{
#if HAVE(PASSKIT_MODULARIZATION) && HAVE(PASSKIT_MAC_HELPER_TEMP)
    if (!PAL::isPassKitMacHelperTempFrameworkAvailable())
#elif HAVE(PASSKIT_MODULARIZATION)
    if (!PAL::isPassKitMacHelperFrameworkAvailable())
#else
    if (!PAL::isPassKitCoreFrameworkAvailable())
#endif
        return completionHandler(false);

    protectedCanMakePaymentsQueue()->dispatch([theClass = retainPtr(PAL::getPKPaymentAuthorizationViewControllerClass()), completionHandler = WTFMove(completionHandler)]() mutable {
        RunLoop::protectedMain()->dispatch([canMakePayments = [theClass canMakePayments], completionHandler = WTFMove(completionHandler)]() mutable {
            completionHandler(canMakePayments);
        });
    });
}

void WebPaymentCoordinatorProxy::platformShowPaymentUI(WebPageProxyIdentifier webPageProxyID, const URL& originatingURL, const Vector<URL>& linkIconURLStrings, const WebCore::ApplePaySessionPaymentRequest& request, CompletionHandler<void(bool)>&& completionHandler)
{
#if HAVE(PASSKIT_MODULARIZATION) && HAVE(PASSKIT_MAC_HELPER_TEMP)
    if (!PAL::isPassKitMacHelperTempFrameworkAvailable())
#elif HAVE(PASSKIT_MODULARIZATION)
    if (!PAL::isPassKitMacHelperFrameworkAvailable())
#else
    if (!PAL::isPassKitCoreFrameworkAvailable())
#endif
        return completionHandler(false);

    RetainPtr<PKPaymentRequest> paymentRequest;
#if HAVE(PASSKIT_DISBURSEMENTS)
    std::optional<ApplePayDisbursementRequest> webDisbursementRequest = request.disbursementRequest();
    if (webDisbursementRequest) {
        auto disbursementRequest = platformDisbursementRequest(request, originatingURL, webDisbursementRequest->requiredRecipientContactFields);
        paymentRequest = RetainPtr<PKPaymentRequest>((PKPaymentRequest *)disbursementRequest.get());
    } else
#endif
        paymentRequest = platformPaymentRequest(originatingURL, linkIconURLStrings, request);

    checkedClient()->getPaymentCoordinatorEmbeddingUserAgent(webPageProxyID, [weakThis = WeakPtr { *this }, paymentRequest, completionHandler = WTFMove(completionHandler)](const String& userAgent) mutable {
        RefPtr paymentCoordinatorProxy = weakThis.get();
        if (!paymentCoordinatorProxy)
            return completionHandler(false);

        paymentCoordinatorProxy->platformSetPaymentRequestUserAgent(paymentRequest.get(), userAgent);

        auto showPaymentUIRequestSeed = weakThis->m_showPaymentUIRequestSeed;

        [PAL::getPKPaymentAuthorizationViewControllerClass() requestViewControllerWithPaymentRequest:paymentRequest.get() completion:makeBlockPtr([paymentRequest, showPaymentUIRequestSeed, weakThis = WTFMove(weakThis), completionHandler = WTFMove(completionHandler)](PKPaymentAuthorizationViewController *viewController, NSError *error) mutable {
            RefPtr paymentCoordinatorProxy = weakThis.get();
            if (!paymentCoordinatorProxy)
                return completionHandler(false);

            if (error) {
                LOG_ERROR("+[PKPaymentAuthorizationViewController requestViewControllerWithPaymentRequest:completion:] error %@", error);

                completionHandler(false);
                return;
            }

            // We've already been asked to hide the payment UI. Don't attempt to show it.
            if (showPaymentUIRequestSeed != paymentCoordinatorProxy->m_showPaymentUIRequestSeed)
                return completionHandler(false);

            NSWindow *presentingWindow = paymentCoordinatorProxy->checkedClient()->paymentCoordinatorPresentingWindow(*paymentCoordinatorProxy);
            if (!presentingWindow)
                return completionHandler(false);

            ASSERT(viewController);

            paymentCoordinatorProxy->m_authorizationPresenter = PaymentAuthorizationViewController::create(*paymentCoordinatorProxy, paymentRequest.get(), viewController);

            ASSERT(!paymentCoordinatorProxy->m_sheetWindow);
            paymentCoordinatorProxy->m_sheetWindow = [NSWindow windowWithContentViewController:viewController];

            paymentCoordinatorProxy->m_sheetWindowWillCloseObserver = [[NSNotificationCenter defaultCenter] addObserverForName:NSWindowWillCloseNotification object:paymentCoordinatorProxy->m_sheetWindow.get() queue:nil usingBlock:[paymentCoordinatorProxy](NSNotification *) {
                paymentCoordinatorProxy->didReachFinalState();
            }];

            [presentingWindow beginSheet:paymentCoordinatorProxy->m_sheetWindow.get() completionHandler:nullptr];

            completionHandler(true);
        }).get()];
    });
}

void WebPaymentCoordinatorProxy::platformHidePaymentUI()
{
    if (m_state == State::Activating) {
        ++m_showPaymentUIRequestSeed;

        ASSERT(!m_authorizationPresenter);
        ASSERT(!m_sheetWindow);
        return;
    }

    ASSERT(m_authorizationPresenter);
    ASSERT(m_sheetWindow);

    [[NSNotificationCenter defaultCenter] removeObserver:m_sheetWindowWillCloseObserver.get()];
    m_sheetWindowWillCloseObserver = nullptr;

    [[m_sheetWindow sheetParent] endSheet:m_sheetWindow.get()];

    if (RefPtr authorizationPresenter = m_authorizationPresenter)
        authorizationPresenter->dismiss();

    m_sheetWindow = nullptr;
}

}

#endif
