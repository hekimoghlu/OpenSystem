/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
#include "WebStorageNamespaceProvider.h"

#include "StorageNamespaceImpl.h"
#include <WebCore/Page.h>
#include <wtf/NeverDestroyed.h>

using namespace WebCore;

namespace WebKit {

static HashSet<WeakRef<WebStorageNamespaceProvider>>& storageNamespaceProviders()
{
    static NeverDestroyed<HashSet<WeakRef<WebStorageNamespaceProvider>>> storageNamespaceProviders;

    return storageNamespaceProviders;
}

Ref<WebStorageNamespaceProvider> WebStorageNamespaceProvider::create(const String& localStorageDatabasePath)
{
    return adoptRef(*new WebStorageNamespaceProvider(localStorageDatabasePath));
}

WebStorageNamespaceProvider::WebStorageNamespaceProvider(const String& localStorageDatabasePath)
    : m_localStorageDatabasePath(localStorageDatabasePath.isNull() ? emptyString() : localStorageDatabasePath)
{
    storageNamespaceProviders().add(*this);
}

WebStorageNamespaceProvider::~WebStorageNamespaceProvider()
{
    ASSERT(storageNamespaceProviders().contains(*this));
    storageNamespaceProviders().remove(*this);
}

void WebStorageNamespaceProvider::closeLocalStorage()
{
    for (const auto& storageNamespaceProvider : storageNamespaceProviders()) {
        if (RefPtr localStorageNamespace = storageNamespaceProvider->optionalLocalStorageNamespace())
            static_cast<StorageNamespaceImpl&>(*localStorageNamespace).close();
    }
}

void WebStorageNamespaceProvider::clearLocalStorageForAllOrigins()
{
    for (const auto& storageNamespaceProvider : storageNamespaceProviders()) {
        if (RefPtr localStorageNamespace = storageNamespaceProvider->optionalLocalStorageNamespace())
            static_cast<StorageNamespaceImpl&>(*localStorageNamespace).clearAllOriginsForDeletion();
    }
}

void WebStorageNamespaceProvider::clearLocalStorageForOrigin(const SecurityOriginData& origin)
{
    for (const auto& storageNamespaceProvider : storageNamespaceProviders()) {
        if (RefPtr localStorageNamespace = storageNamespaceProvider->optionalLocalStorageNamespace())
            static_cast<StorageNamespaceImpl&>(*localStorageNamespace).clearOriginForDeletion(origin);
    }
}

void WebStorageNamespaceProvider::closeIdleLocalStorageDatabases()
{
    for (const auto& storageNamespaceProvider : storageNamespaceProviders()) {
        if (RefPtr localStorageNamespace = storageNamespaceProvider->optionalLocalStorageNamespace())
            static_cast<StorageNamespaceImpl&>(*localStorageNamespace).closeIdleLocalStorageDatabases();
    }
}

void WebStorageNamespaceProvider::syncLocalStorage()
{
    for (const auto& storageNamespaceProvider : storageNamespaceProviders()) {
        if (RefPtr localStorageNamespace = storageNamespaceProvider->optionalLocalStorageNamespace())
            static_cast<StorageNamespaceImpl&>(*localStorageNamespace).sync();
    }
}

Ref<StorageNamespace> WebStorageNamespaceProvider::createLocalStorageNamespace(unsigned quota, PAL::SessionID sessionID)
{
    return StorageNamespaceImpl::getOrCreateLocalStorageNamespace(m_localStorageDatabasePath, quota, sessionID);
}

Ref<StorageNamespace> WebStorageNamespaceProvider::createTransientLocalStorageNamespace(SecurityOrigin&, unsigned quota, PAL::SessionID sessionID)
{
    // FIXME: A smarter implementation would create a special namespace type instead of just piggy-backing off
    // SessionStorageNamespace here.
    return StorageNamespaceImpl::createSessionStorageNamespace(quota, sessionID);
}

RefPtr<StorageNamespace> WebStorageNamespaceProvider::sessionStorageNamespace(const SecurityOrigin& topLevelOrigin, Page& page, ShouldCreateNamespace shouldCreate)
{
    ASSERT(sessionStorageQuota() != WebCore::StorageMap::noQuota);

    if (m_sessionStorageNamespaces.find(page) == m_sessionStorageNamespaces.end()) {
        if (shouldCreate == ShouldCreateNamespace::No)
            return nullptr;
        HashMap<SecurityOriginData, RefPtr<StorageNamespace>> map;
        m_sessionStorageNamespaces.set(page, map);
    }
    auto& sessionStorageNamespaces = m_sessionStorageNamespaces.find(page)->value;

    auto sessionStorageNamespaceIt = sessionStorageNamespaces.find(topLevelOrigin.data());
    if (sessionStorageNamespaceIt == sessionStorageNamespaces.end()) {
        if (shouldCreate == ShouldCreateNamespace::No)
            return nullptr;
        return sessionStorageNamespaces.add(topLevelOrigin.data(), StorageNamespaceImpl::createSessionStorageNamespace(sessionStorageQuota(), page.sessionID())).iterator->value;
    }
    return sessionStorageNamespaceIt->value;
}

void WebStorageNamespaceProvider::cloneSessionStorageNamespaceForPage(WebCore::Page& srcPage, WebCore::Page& dstPage)
{
    ASSERT(sessionStorageQuota() != WebCore::StorageMap::noQuota);

    auto& srcSessionStorageNamespaces = static_cast<WebStorageNamespaceProvider&>(srcPage.storageNamespaceProvider()).m_sessionStorageNamespaces;
    auto srcPageIt = srcSessionStorageNamespaces.find(srcPage);
    if (srcPageIt == srcSessionStorageNamespaces.end())
        return;

    auto& srcPageSessionStorageNamespaces = srcPageIt->value;
    HashMap<SecurityOriginData, RefPtr<StorageNamespace>> dstPageSessionStorageNamespaces;
    for (auto& [origin, srcNamespace] : srcPageSessionStorageNamespaces)
        dstPageSessionStorageNamespaces.set(origin, srcNamespace->copy(dstPage));

    auto& dstSessionStorageNamespaces = static_cast<WebStorageNamespaceProvider&>(dstPage.storageNamespaceProvider()).m_sessionStorageNamespaces;
    ASSERT(!dstSessionStorageNamespaces.contains(dstPage));
    dstSessionStorageNamespaces.set(dstPage, WTFMove(dstPageSessionStorageNamespaces));
}

}
