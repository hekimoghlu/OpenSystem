/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

#if ENABLE(WEB_AUTHN)

#include "AuthenticatorManager.h"
#include <WebCore/MockWebAuthenticationConfiguration.h>

namespace WebKit {

class MockAuthenticatorManager final : public AuthenticatorManager {
    WTF_MAKE_TZONE_ALLOCATED(MockAuthenticatorManager);
public:
    static Ref<MockAuthenticatorManager> create(WebCore::MockWebAuthenticationConfiguration&&);

    bool isMock() const final { return true; }
    void setTestConfiguration(WebCore::MockWebAuthenticationConfiguration&& configuration) { m_testConfiguration = WTFMove(configuration); }

private:
    explicit MockAuthenticatorManager(WebCore::MockWebAuthenticationConfiguration&&);

    Ref<AuthenticatorTransportService> createService(WebCore::AuthenticatorTransport, AuthenticatorTransportServiceObserver&) const final;
    void respondReceivedInternal(Respond&&) final;
    void filterTransports(TransportSet&) const;
    void runPresenterInternal(const TransportSet&) final { }

    WebCore::MockWebAuthenticationConfiguration m_testConfiguration;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::MockAuthenticatorManager)
static bool isType(const WebKit::AuthenticatorManager& manager) { return manager.isMock(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_AUTHN)
