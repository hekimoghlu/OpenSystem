/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include "MockAuthenticatorManager.h"

#if ENABLE(WEB_AUTHN)

#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MockAuthenticatorManager);

Ref<MockAuthenticatorManager> MockAuthenticatorManager::create(WebCore::MockWebAuthenticationConfiguration&& configuration)
{
    return adoptRef(*new MockAuthenticatorManager(WTFMove(configuration)));
}

MockAuthenticatorManager::MockAuthenticatorManager(WebCore::MockWebAuthenticationConfiguration&& configuration)
    : m_testConfiguration(WTFMove(configuration))
{
}

Ref<AuthenticatorTransportService> MockAuthenticatorManager::createService(WebCore::AuthenticatorTransport transport, AuthenticatorTransportServiceObserver& observer) const
{
    return AuthenticatorTransportService::createMock(transport, observer, m_testConfiguration);
}

void MockAuthenticatorManager::respondReceivedInternal(Respond&& respond)
{
    if (m_testConfiguration.silentFailure)
        return;

    invokePendingCompletionHandler(WTFMove(respond));
    clearStateAsync();
    requestTimeOutTimer().stop();
}

void MockAuthenticatorManager::filterTransports(TransportSet& transports) const
{
    if (!m_testConfiguration.nfc)
        transports.remove(WebCore::AuthenticatorTransport::Nfc);
    if (!m_testConfiguration.local)
        transports.remove(WebCore::AuthenticatorTransport::Internal);
    if (!m_testConfiguration.ccid)
        transports.remove(WebCore::AuthenticatorTransport::SmartCard);
    transports.remove(WebCore::AuthenticatorTransport::Ble);
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
