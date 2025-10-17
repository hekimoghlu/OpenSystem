/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#include "VirtualAuthenticatorManager.h"

#if ENABLE(WEB_AUTHN)

#include <VirtualService.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UUID.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VirtualAuthenticatorManager);

struct VirtualCredential;

Ref<VirtualAuthenticatorManager> VirtualAuthenticatorManager::create()
{
    return adoptRef(*new VirtualAuthenticatorManager);
}

VirtualAuthenticatorManager::VirtualAuthenticatorManager() = default;

String VirtualAuthenticatorManager::createAuthenticator(const VirtualAuthenticatorConfiguration& config)
{
    auto id = createVersion4UUIDString();
    m_virtualAuthenticators.set(id, makeUniqueRef<VirtualAuthenticatorConfiguration>(config));
    Vector<VirtualCredential> credentials;
    m_credentialsByAuthenticator.set(id, WTFMove(credentials));

    return id;
}

bool VirtualAuthenticatorManager::removeAuthenticator(const String& id)
{
    return m_virtualAuthenticators.remove(id);
}

void VirtualAuthenticatorManager::addCredential(const String& authenticatorId, VirtualCredential& credential)
{
    m_credentialsByAuthenticator.find(authenticatorId)->value.append(WTFMove(credential));
}

Vector<VirtualCredential> VirtualAuthenticatorManager::credentialsMatchingList(const String& authenticatorId, const String& rpId, const Vector<Vector<uint8_t>>& credentialIds)
{
    Vector<VirtualCredential> matching;
    auto it = m_credentialsByAuthenticator.find(authenticatorId);
    for (auto& credential : it->value) {
        if (credential.rpId == rpId && ((credentialIds.isEmpty() && credential.isResidentCredential) || credentialIds.contains(credential.credentialId)))
            matching.append(credential);
    }
    return matching;
}

Ref<AuthenticatorTransportService> VirtualAuthenticatorManager::createService(WebCore::AuthenticatorTransport transport, AuthenticatorTransportServiceObserver& observer) const
{
    Vector<std::pair<String, VirtualAuthenticatorConfiguration>> configs;
    for (auto& id : m_virtualAuthenticators.keys()) {
        auto config = m_virtualAuthenticators.get(id);
        if (config->transport == transport)
            configs.append(std::pair { id, *config });
    }
    return VirtualService::createVirtual(transport, observer, configs);
}

void VirtualAuthenticatorManager::runPanel()
{
    auto transports = getTransports();
    if (transports.isEmpty()) {
        cancel();
        return;
    }

    startDiscovery(transports);
}

void VirtualAuthenticatorManager::selectAssertionResponse(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&& responses, WebAuthenticationSource source, CompletionHandler<void(WebCore::AuthenticatorAssertionResponse*)>&& completionHandler)
{
    completionHandler(responses[0].ptr());
}

void VirtualAuthenticatorManager::decidePolicyForLocalAuthenticator(CompletionHandler<void(LocalAuthenticatorPolicy)>&& completionHandler)
{
    completionHandler(LocalAuthenticatorPolicy::Allow);
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
