/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#include "VirtualAuthenticatorConfiguration.h"
#include "VirtualCredential.h"

namespace WebKit {
struct VirtualCredential;

class VirtualAuthenticatorManager final : public AuthenticatorManager {
    WTF_MAKE_TZONE_ALLOCATED(VirtualAuthenticatorManager);
public:
    static Ref<VirtualAuthenticatorManager> create();

    String createAuthenticator(const VirtualAuthenticatorConfiguration& /*config/*/);
    bool removeAuthenticator(const String& /*authenticatorId*/);

    bool isVirtual() const final { return true; }

    void addCredential(const String&, VirtualCredential&);
    Vector<VirtualCredential> credentialsMatchingList(const String& authenticatorId, const String& rpId, const Vector<Vector<uint8_t>>& credentialIds);

protected:
    void decidePolicyForLocalAuthenticator(CompletionHandler<void(LocalAuthenticatorPolicy)>&&) override;
    void selectAssertionResponse(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&&, WebAuthenticationSource, CompletionHandler<void(WebCore::AuthenticatorAssertionResponse*)>&&) override;
    
    
private:
    VirtualAuthenticatorManager();

    Ref<AuthenticatorTransportService> createService(WebCore::AuthenticatorTransport, AuthenticatorTransportServiceObserver&) const final;
    void runPanel() override;
    void filterTransports(TransportSet&) const override { };

    HashMap<String, UniqueRef<VirtualAuthenticatorConfiguration>> m_virtualAuthenticators;
    HashMap<String, Vector<VirtualCredential>> m_credentialsByAuthenticator;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::VirtualAuthenticatorManager)
static bool isType(const WebKit::AuthenticatorManager& manager) { return manager.isVirtual(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_AUTHN)
