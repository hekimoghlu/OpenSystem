/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#include "NetworkContentRuleListManager.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include "NetworkProcess.h"
#include "NetworkProcessProxyMessages.h"
#include "WebCompiledContentRuleList.h"

namespace WebKit {
using namespace WebCore;

NetworkContentRuleListManager::NetworkContentRuleListManager(NetworkProcess& networkProcess)
    : m_networkProcess(networkProcess)
{
}

NetworkContentRuleListManager::~NetworkContentRuleListManager()
{
    auto pendingCallbacks = WTFMove(m_pendingCallbacks);
    if (pendingCallbacks.isEmpty())
        return;

    WebCore::ContentExtensions::ContentExtensionsBackend backend;
    for (auto& callbacks : pendingCallbacks.values()) {
        for (auto& callback : callbacks)
            callback(backend);
    }
}

void NetworkContentRuleListManager::ref() const
{
    m_networkProcess->ref();
}

void NetworkContentRuleListManager::deref() const
{
    m_networkProcess->deref();
}

Ref<NetworkProcess> NetworkContentRuleListManager::protectedNetworkProcess() const
{
    ASSERT(RunLoop::isMain());
    return m_networkProcess.get();
}

void NetworkContentRuleListManager::contentExtensionsBackend(UserContentControllerIdentifier identifier, BackendCallback&& callback)
{
    auto iterator = m_contentExtensionBackends.find(identifier);
    if (iterator != m_contentExtensionBackends.end()) {
        callback(*iterator->value);
        return;
    }
    m_pendingCallbacks.ensure(identifier, [] {
        return Vector<BackendCallback> { };
    }).iterator->value.append(WTFMove(callback));
    protectedNetworkProcess()->protectedParentProcessConnection()->send(Messages::NetworkProcessProxy::ContentExtensionRules { identifier }, 0);
}

void NetworkContentRuleListManager::addContentRuleLists(UserContentControllerIdentifier identifier, Vector<std::pair<WebCompiledContentRuleListData, URL>>&& contentRuleLists)
{
    auto& backend = *m_contentExtensionBackends.ensure(identifier, [] {
        return makeUnique<WebCore::ContentExtensions::ContentExtensionsBackend>();
    }).iterator->value;

    for (auto&& pair : contentRuleLists) {
        auto&& contentRuleList = WTFMove(pair.first);
        String identifier = contentRuleList.identifier;
        if (RefPtr compiledContentRuleList = WebCompiledContentRuleList::create(WTFMove(contentRuleList)))
            backend.addContentExtension(identifier, compiledContentRuleList.releaseNonNull(), WTFMove(pair.second), ContentExtensions::ContentExtension::ShouldCompileCSS::No);
    }

    auto pendingCallbacks = m_pendingCallbacks.take(identifier);
    for (auto& callback : pendingCallbacks)
        callback(backend);

}

void NetworkContentRuleListManager::removeContentRuleList(UserContentControllerIdentifier identifier, const String& name)
{
    auto iterator = m_contentExtensionBackends.find(identifier);
    if (iterator == m_contentExtensionBackends.end())
        return;

    iterator->value->removeContentExtension(name);
}

void NetworkContentRuleListManager::removeAllContentRuleLists(UserContentControllerIdentifier identifier)
{
    auto iterator = m_contentExtensionBackends.find(identifier);
    if (iterator == m_contentExtensionBackends.end())
        return;

    iterator->value->removeAllContentExtensions();
}

void NetworkContentRuleListManager::remove(UserContentControllerIdentifier identifier)
{
    m_contentExtensionBackends.remove(identifier);
}

} // namespace WebKit

#endif // ENABLE(CONTENT_EXTENSIONS)
