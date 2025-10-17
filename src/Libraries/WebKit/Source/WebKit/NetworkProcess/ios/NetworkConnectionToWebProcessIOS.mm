/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#import "NetworkConnectionToWebProcess.h"
#import "NetworkProcessProxyMessages.h"

#if PLATFORM(IOS_FAMILY)

#import "NetworkProcess.h"
#import "NetworkSessionCocoa.h"
#import "PaymentAuthorizationController.h"

namespace WebKit {
    
#if ENABLE(APPLE_PAY_REMOTE_UI)

WebPaymentCoordinatorProxy& NetworkConnectionToWebProcess::paymentCoordinator()
{
    if (!m_paymentCoordinator)
        m_paymentCoordinator = WebPaymentCoordinatorProxy::create(*this);
    return *m_paymentCoordinator;
}

IPC::Connection* NetworkConnectionToWebProcess::paymentCoordinatorConnection(const WebPaymentCoordinatorProxy&)
{
    return &connection();
}

UIViewController *NetworkConnectionToWebProcess::paymentCoordinatorPresentingViewController(const WebPaymentCoordinatorProxy&)
{
    return nil;
}

#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
void NetworkConnectionToWebProcess::getWindowSceneAndBundleIdentifierForPaymentPresentation(WebPageProxyIdentifier webPageProxyIdentifier, CompletionHandler<void(const String&, const String&)>&& completionHandler)
{
    networkProcess().parentProcessConnection()->sendWithAsyncReply(Messages::NetworkProcessProxy::GetWindowSceneAndBundleIdentifierForPaymentPresentation(webPageProxyIdentifier), WTFMove(completionHandler));
}
#endif

void NetworkConnectionToWebProcess::getPaymentCoordinatorEmbeddingUserAgent(WebPageProxyIdentifier webPageProxyIdentifier, CompletionHandler<void(const String&)>&& completionHandler)
{
    networkProcess().parentProcessConnection()->sendWithAsyncReply(Messages::NetworkProcessProxy::GetPaymentCoordinatorEmbeddingUserAgent { webPageProxyIdentifier }, WTFMove(completionHandler));
}

CocoaWindow *NetworkConnectionToWebProcess::paymentCoordinatorPresentingWindow(const WebPaymentCoordinatorProxy&) const
{
    return nil;
}

std::optional<SharedPreferencesForWebProcess> NetworkConnectionToWebProcess::sharedPreferencesForWebPaymentMessages() const
{
    return m_sharedPreferencesForWebProcess;
}

const String& NetworkConnectionToWebProcess::paymentCoordinatorBoundInterfaceIdentifier(const WebPaymentCoordinatorProxy&)
{
    if (auto* session = static_cast<NetworkSessionCocoa*>(networkSession()))
        return session->boundInterfaceIdentifier();
    return emptyString();
}

const String& NetworkConnectionToWebProcess::paymentCoordinatorCTDataConnectionServiceType(const WebPaymentCoordinatorProxy&)
{
    if (auto* session = static_cast<NetworkSessionCocoa*>(networkSession()))
        return session->dataConnectionServiceType();
    return emptyString();
}

const String& NetworkConnectionToWebProcess::paymentCoordinatorSourceApplicationBundleIdentifier(const WebPaymentCoordinatorProxy&)
{
    if (auto* session = static_cast<NetworkSessionCocoa*>(networkSession()))
        return session->sourceApplicationBundleIdentifier();
    return emptyString();
}

const String& NetworkConnectionToWebProcess::paymentCoordinatorSourceApplicationSecondaryIdentifier(const WebPaymentCoordinatorProxy&)
{
    if (auto* session = static_cast<NetworkSessionCocoa*>(networkSession()))
        return session->sourceApplicationSecondaryIdentifier();
    return emptyString();
}

Ref<PaymentAuthorizationPresenter> NetworkConnectionToWebProcess::paymentCoordinatorAuthorizationPresenter(WebPaymentCoordinatorProxy& coordinator, PKPaymentRequest *request)
{
    return PaymentAuthorizationController::create(coordinator, request);
}

void NetworkConnectionToWebProcess::paymentCoordinatorAddMessageReceiver(WebPaymentCoordinatorProxy&, IPC::ReceiverName, IPC::MessageReceiver&)
{
}

void NetworkConnectionToWebProcess::paymentCoordinatorRemoveMessageReceiver(WebPaymentCoordinatorProxy&, IPC::ReceiverName)
{
}

#endif // ENABLE(APPLE_PAY_REMOTE_UI)

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
