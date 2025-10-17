/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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

#if PLATFORM(IOS_FAMILY) && ENABLE(APPLE_PAY)

#import "APIUIClient.h"
#import "PaymentAuthorizationPresenter.h"
#import "WebPageProxy.h"
#import <UIKit/UIViewController.h>
#import <pal/cocoa/PassKitSoftLink.h>

namespace WebKit {

void WebPaymentCoordinatorProxy::platformCanMakePayments(CompletionHandler<void(bool)>&& completionHandler)
{
    m_canMakePaymentsQueue->dispatch([theClass = retainPtr(PAL::getPKPaymentAuthorizationControllerClass()), completionHandler = WTFMove(completionHandler)]() mutable {
        RunLoop::main().dispatch([canMakePayments = [theClass canMakePayments], completionHandler = WTFMove(completionHandler)]() mutable {
            completionHandler(canMakePayments);
        });
    });
}

void WebPaymentCoordinatorProxy::platformShowPaymentUI(WebPageProxyIdentifier webPageProxyID, const URL& originatingURL, const Vector<URL>& linkIconURLStrings, const WebCore::ApplePaySessionPaymentRequest& request, CompletionHandler<void(bool)>&& completionHandler)
{

    RetainPtr<PKPaymentRequest> paymentRequest;
#if HAVE(PASSKIT_DISBURSEMENTS)
    std::optional<ApplePayDisbursementRequest> webDisbursementRequest = request.disbursementRequest();
    if (webDisbursementRequest) {
        auto disbursementRequest = platformDisbursementRequest(request, originatingURL, webDisbursementRequest->requiredRecipientContactFields);
        paymentRequest = RetainPtr<PKPaymentRequest>((PKPaymentRequest *)disbursementRequest.get());
    } else
#endif
        paymentRequest = platformPaymentRequest(originatingURL, linkIconURLStrings, request);

    checkedClient()->getPaymentCoordinatorEmbeddingUserAgent(webPageProxyID, [webPageProxyID, paymentRequest, weakThis = WeakPtr { *this }, completionHandler = WTFMove(completionHandler)](const String& userAgent) mutable {
        auto paymentCoordinatorProxy = weakThis.get();
        if (!paymentCoordinatorProxy)
            return completionHandler(false);

        paymentCoordinatorProxy->platformSetPaymentRequestUserAgent(paymentRequest.get(), userAgent);

        ASSERT(!paymentCoordinatorProxy->m_authorizationPresenter);
        paymentCoordinatorProxy->m_authorizationPresenter = paymentCoordinatorProxy->checkedClient()->paymentCoordinatorAuthorizationPresenter(*paymentCoordinatorProxy, paymentRequest.get());
        if (!paymentCoordinatorProxy->m_authorizationPresenter)
            return completionHandler(false);

#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
        paymentCoordinatorProxy->checkedClient()->getWindowSceneAndBundleIdentifierForPaymentPresentation(webPageProxyID, [weakThis = WTFMove(weakThis), completionHandler = WTFMove(completionHandler)](const String& sceneIdentifier, const String& bundleIdentifier) mutable {
            auto paymentCoordinatorProxy = weakThis.get();
            if (!paymentCoordinatorProxy)
                return completionHandler(false);

            if (!paymentCoordinatorProxy->m_authorizationPresenter)
                return completionHandler(false);

            paymentCoordinatorProxy->m_authorizationPresenter->presentInScene(sceneIdentifier, bundleIdentifier, WTFMove(completionHandler));
        });
#else
        UNUSED_VARIABLE(webPageProxyID);
        paymentCoordinatorProxy->m_authorizationPresenter->present(paymentCoordinatorProxy->checkedClient()->paymentCoordinatorPresentingViewController(*paymentCoordinatorProxy), WTFMove(completionHandler));
#endif
    });
}

void WebPaymentCoordinatorProxy::platformHidePaymentUI()
{
    if (m_authorizationPresenter)
        m_authorizationPresenter->dismiss();
}

}

#endif
