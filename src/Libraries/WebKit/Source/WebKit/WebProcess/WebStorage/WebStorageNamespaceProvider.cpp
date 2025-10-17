/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
#include "WebStorageNamespaceProvider.h"

#include "NetworkProcessConnection.h"
#include "NetworkStorageManagerMessages.h"
#include "WebPage.h"
#include "WebPageInlines.h"
#include "WebProcess.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/WeakPtr.h>

namespace WebKit {
using namespace WebCore;

static WeakPtr<WebStorageNamespaceProvider>& existingStorageNameSpaceProvider()
{
    static NeverDestroyed<WeakPtr<WebStorageNamespaceProvider>> storageNameSpaceProvider;
    return storageNameSpaceProvider.get();
}

Ref<WebStorageNamespaceProvider> WebStorageNamespaceProvider::getOrCreate()
{
    if (auto& provider = existingStorageNameSpaceProvider())
        return Ref { *provider.get() };

    return adoptRef(*new WebStorageNamespaceProvider());
}

void WebStorageNamespaceProvider::incrementUseCount(const StorageNamespaceImpl::Identifier identifier)
{
    if (auto& provider = existingStorageNameSpaceProvider()) {
        auto& sessionStorageNamespaces = provider->m_sessionStorageNamespaces.add(identifier, SessionStorageNamespaces { }).iterator->value;
        ++sessionStorageNamespaces.useCount;
    }
}

void WebStorageNamespaceProvider::decrementUseCount(const StorageNamespaceImpl::Identifier identifier)
{
    if (auto& provider = existingStorageNameSpaceProvider()) {
        auto iterator = provider->m_sessionStorageNamespaces.find(identifier);
        ASSERT(iterator != provider->m_sessionStorageNamespaces.end());
        auto& sessionStorageNamespaces = iterator->value;
        ASSERT(sessionStorageNamespaces.useCount);
        if (!--sessionStorageNamespaces.useCount)
            provider->m_sessionStorageNamespaces.remove(identifier);
    }
}

WebStorageNamespaceProvider::WebStorageNamespaceProvider()
{
    existingStorageNameSpaceProvider() = *this;
}

WebStorageNamespaceProvider::~WebStorageNamespaceProvider() = default;

Ref<WebCore::StorageNamespace> WebStorageNamespaceProvider::createLocalStorageNamespace(unsigned quota, PAL::SessionID sessionID)
{
    ASSERT_UNUSED(sessionID, sessionID == WebProcess::singleton().sessionID());
    return StorageNamespaceImpl::createLocalStorageNamespace(quota);
}

Ref<WebCore::StorageNamespace> WebStorageNamespaceProvider::createTransientLocalStorageNamespace(WebCore::SecurityOrigin& topLevelOrigin, unsigned quota, PAL::SessionID sessionID)
{
    ASSERT_UNUSED(sessionID, sessionID == WebProcess::singleton().sessionID());
    return StorageNamespaceImpl::createTransientLocalStorageNamespace(topLevelOrigin, quota);
}

RefPtr<WebCore::StorageNamespace> WebStorageNamespaceProvider::sessionStorageNamespace(const WebCore::SecurityOrigin& topLevelOrigin, WebCore::Page& page, ShouldCreateNamespace shouldCreate)
{
    ASSERT(sessionStorageQuota() != WebCore::StorageMap::noQuota);

    RefPtr webPage = WebPage::fromCorePage(page);
    if (!webPage)
        return nullptr;

    // The identifier of a session storage namespace is the WebPageProxyIdentifier. It is possible we have several WebPage objects in a single process for the same
    // WebPageProxyIdentifier and these need to share the same namespace instance so we know where to route the IPC to.
    auto namespacesIt = m_sessionStorageNamespaces.find(webPage->sessionStorageNamespaceIdentifier());
    if (namespacesIt == m_sessionStorageNamespaces.end()) {
        if (shouldCreate == ShouldCreateNamespace::No)
            return nullptr;
        namespacesIt = m_sessionStorageNamespaces.set(webPage->sessionStorageNamespaceIdentifier(), SessionStorageNamespaces { }).iterator;
    }

    auto& sessionStorageNamespacesMap = namespacesIt->value.map;
    auto it = sessionStorageNamespacesMap.find(topLevelOrigin.data());
    if (it == sessionStorageNamespacesMap.end()) {
        if (shouldCreate == ShouldCreateNamespace::No)
            return nullptr;
        auto sessionStorageNamespace = StorageNamespaceImpl::createSessionStorageNamespace(webPage->sessionStorageNamespaceIdentifier(), webPage->identifier(), topLevelOrigin, sessionStorageQuota());
        it = sessionStorageNamespacesMap.set(topLevelOrigin.data(), WTFMove(sessionStorageNamespace)).iterator;
    }
    return it->value;
}

}
