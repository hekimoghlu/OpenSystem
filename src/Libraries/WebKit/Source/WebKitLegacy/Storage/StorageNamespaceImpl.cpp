/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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
#include "StorageNamespaceImpl.h"

#include "StorageAreaImpl.h"
#include "StorageSyncManager.h"
#include "StorageTracker.h"
#include <WebCore/SecurityOrigin.h>
#include <WebCore/StorageMap.h>
#include <WebCore/StorageType.h>
#include <wtf/CheckedPtr.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringHash.h>

using namespace WebCore;

namespace WebKit {

static HashMap<String, WeakRef<StorageNamespaceImpl>>& localStorageNamespaceMap()
{
    static NeverDestroyed<HashMap<String, WeakRef<StorageNamespaceImpl>>> localStorageNamespaceMap;

    return localStorageNamespaceMap;
}

Ref<StorageNamespaceImpl> StorageNamespaceImpl::createSessionStorageNamespace(unsigned quota, PAL::SessionID sessionID)
{
    return adoptRef(*new StorageNamespaceImpl(StorageType::Session, String(), quota, sessionID));
}

Ref<StorageNamespaceImpl> StorageNamespaceImpl::getOrCreateLocalStorageNamespace(const String& databasePath, unsigned quota, PAL::SessionID sessionID)
{
    ASSERT(!databasePath.isNull());

    RefPtr<StorageNamespaceImpl> storageNamespace;
    auto& slot = localStorageNamespaceMap().ensure(databasePath, [&] {
        storageNamespace = adoptRef(*new StorageNamespaceImpl(StorageType::Local, databasePath, quota, sessionID));
        return WeakRef { *storageNamespace };
    }).iterator->value;
    return storageNamespace ? storageNamespace.releaseNonNull() : Ref { slot.get() };
}

StorageNamespaceImpl::StorageNamespaceImpl(StorageType storageType, const String& path, unsigned quota, PAL::SessionID sessionID)
    : m_storageType(storageType)
    , m_path(path.isolatedCopy())
    , m_syncManager(nullptr)
    , m_quota(quota)
    , m_isShutdown(false)
    , m_sessionID(sessionID)
{
    if (isLocalStorage(m_storageType) && !m_path.isEmpty())
        m_syncManager = StorageSyncManager::create(m_path);
}

StorageNamespaceImpl::~StorageNamespaceImpl()
{
    ASSERT(isMainThread());

    if (isLocalStorage(m_storageType)) {
        ASSERT(localStorageNamespaceMap().get(m_path) == this);
        localStorageNamespaceMap().remove(m_path);
    }

    if (!m_isShutdown)
        close();
}

Ref<StorageNamespace> StorageNamespaceImpl::copy(Page&)
{
    ASSERT(isMainThread());
    ASSERT(!m_isShutdown);
    ASSERT(m_storageType == StorageType::Session);

    auto newNamespace = adoptRef(*new StorageNamespaceImpl(m_storageType, m_path, m_quota, m_sessionID));
    for (auto& iter : m_storageAreaMap)
        newNamespace->m_storageAreaMap.set(iter.key, iter.value->copy());

    return WTFMove(newNamespace);
}

Ref<StorageArea> StorageNamespaceImpl::storageArea(const SecurityOrigin& origin)
{
    ASSERT(isMainThread());
    ASSERT(!m_isShutdown);

    return *m_storageAreaMap.ensure(origin.data(), [&] {
        return StorageAreaImpl::create(m_storageType, origin, m_syncManager.get(), m_quota);
    }).iterator->value;
}

void StorageNamespaceImpl::close()
{
    ASSERT(isMainThread());

    if (m_isShutdown)
        return;

    // If we're not a persistent storage, we shouldn't need to do any work here.
    if (m_storageType == StorageType::Session) {
        ASSERT(!m_syncManager);
        return;
    }

    StorageAreaMap::iterator end = m_storageAreaMap.end();
    for (StorageAreaMap::iterator it = m_storageAreaMap.begin(); it != end; ++it)
        it->value->close();

    if (m_syncManager)
        m_syncManager->close();

    m_isShutdown = true;
}

void StorageNamespaceImpl::clearOriginForDeletion(const SecurityOriginData& origin)
{
    ASSERT(isMainThread());

    RefPtr<StorageAreaImpl> storageArea = m_storageAreaMap.get(origin);
    if (storageArea)
        storageArea->clearForOriginDeletion();
}

void StorageNamespaceImpl::clearAllOriginsForDeletion()
{
    ASSERT(isMainThread());

    StorageAreaMap::iterator end = m_storageAreaMap.end();
    for (StorageAreaMap::iterator it = m_storageAreaMap.begin(); it != end; ++it)
        it->value->clearForOriginDeletion();
}
    
void StorageNamespaceImpl::sync()
{
    ASSERT(isMainThread());
    StorageAreaMap::iterator end = m_storageAreaMap.end();
    for (StorageAreaMap::iterator it = m_storageAreaMap.begin(); it != end; ++it)
        it->value->sync();
}

void StorageNamespaceImpl::closeIdleLocalStorageDatabases()
{
    ASSERT(isMainThread());
    StorageAreaMap::iterator end = m_storageAreaMap.end();
    for (StorageAreaMap::iterator it = m_storageAreaMap.begin(); it != end; ++it)
        it->value->closeDatabaseIfIdle();
}

void StorageNamespaceImpl::setSessionIDForTesting(PAL::SessionID sessionID)
{
    m_sessionID = sessionID;
    for (auto storageAreaMap : m_storageAreaMap.values())
        storageAreaMap->sessionChanged(!sessionID.isEphemeral());
}

} // namespace WebCore
