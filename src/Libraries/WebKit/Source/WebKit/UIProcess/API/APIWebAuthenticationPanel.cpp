/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#include "APIWebAuthenticationPanel.h"

#if ENABLE(WEB_AUTHN)

#include "APIWebAuthenticationPanelClient.h"
#include "AuthenticatorManager.h"
#include "MockAuthenticatorManager.h"
#include <WebCore/WebAuthenticationConstants.h>

namespace API {
using namespace WebCore;
using namespace WebKit;

Ref<WebAuthenticationPanel> WebAuthenticationPanel::create(const AuthenticatorManager& manager, const WTF::String& rpId, const TransportSet& transports, ClientDataType type, const WTF::String& userName)
{
    return adoptRef(*new WebAuthenticationPanel(manager, rpId, transports, type, userName));
}

WebAuthenticationPanel::WebAuthenticationPanel()
    : m_manager(AuthenticatorManager::create())
    , m_client(WebAuthenticationPanelClient::create())
{
    protectedManager()->enableNativeSupport();
}

RefPtr<WebKit::AuthenticatorManager> WebAuthenticationPanel::protectedManager() const
{
    return m_manager;
}

WebAuthenticationPanel::WebAuthenticationPanel(const AuthenticatorManager& manager, const WTF::String& rpId, const TransportSet& transports, ClientDataType type, const WTF::String& userName)
    : m_client(WebAuthenticationPanelClient::create())
    , m_weakManager(manager)
    , m_rpId(rpId)
    , m_clientDataType(type)
    , m_userName(userName)
{
    m_transports.reserveInitialCapacity(AuthenticatorManager::maxTransportNumber);
    if (transports.contains(AuthenticatorTransport::Usb))
        m_transports.append(AuthenticatorTransport::Usb);
    if (transports.contains(AuthenticatorTransport::Nfc))
        m_transports.append(AuthenticatorTransport::Nfc);
    if (transports.contains(AuthenticatorTransport::Internal))
        m_transports.append(AuthenticatorTransport::Internal);
}

WebAuthenticationPanel::~WebAuthenticationPanel() = default;

void WebAuthenticationPanel::handleRequest(WebAuthenticationRequestData&& request, Callback&& callback)
{
    ASSERT(m_manager);
    request.weakPanel = *this;
    protectedManager()->handleRequest(WTFMove(request), WTFMove(callback));
}

void WebAuthenticationPanel::cancel() const
{
    if (RefPtr manager = m_weakManager.get()) {
        manager->cancelRequest(*this);
        return;
    }

    protectedManager()->cancel();
}

void WebAuthenticationPanel::setMockConfiguration(WebCore::MockWebAuthenticationConfiguration&& configuration)
{
    ASSERT(m_manager);

    if (RefPtr mockManager = dynamicDowncast<MockAuthenticatorManager>(*m_manager)) {
        mockManager->setTestConfiguration(WTFMove(configuration));
        return;
    }

    Ref manager = MockAuthenticatorManager::create(WTFMove(configuration));
    manager->enableNativeSupport();
    m_manager = WTFMove(manager);
}

void WebAuthenticationPanel::setClient(Ref<WebAuthenticationPanelClient>&& client)
{
    m_client = WTFMove(client);
}

} // namespace API

#endif // ENABLE(WEB_AUTHN)
