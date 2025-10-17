/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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
#include "StorageAreaImpl.h"

#include "StorageAreaSync.h"
#include "StorageSyncManager.h"
#include "StorageTracker.h"
#include <WebCore/LocalDOMWindow.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Page.h>
#include <WebCore/SecurityOrigin.h>
#include <WebCore/SecurityOriginData.h>
#include <WebCore/Storage.h>
#include <WebCore/StorageEventDispatcher.h>
#include <WebCore/StorageType.h>
#include <wtf/MainThread.h>

using namespace WebCore;

namespace WebKit {

StorageAreaImpl::~StorageAreaImpl()
{
    ASSERT(isMainThread());
}

inline StorageAreaImpl::StorageAreaImpl(StorageType storageType, const SecurityOrigin& origin, RefPtr<StorageSyncManager>&& syncManager, unsigned quota)
    : m_storageType(storageType)
    , m_securityOrigin(origin)
    , m_storageMap(quota)
    , m_storageSyncManager(WTFMove(syncManager))
    , m_accessCount(0)
    , m_closeDatabaseTimer(*this, &StorageAreaImpl::closeDatabaseTimerFired)
{
    ASSERT(isMainThread());

    // Accessing the shared global StorageTracker when a StorageArea is created
    // ensures that the tracker is properly initialized before anyone actually needs to use it.
    StorageTracker::tracker();
}

Ref<StorageAreaImpl> StorageAreaImpl::create(StorageType storageType, const SecurityOrigin& origin, RefPtr<StorageSyncManager>&& syncManager, unsigned quota)
{
    Ref<StorageAreaImpl> area = adoptRef(*new StorageAreaImpl(storageType, origin, WTFMove(syncManager), quota));
    // FIXME: If there's no backing storage for LocalStorage, the default WebKit behavior should be that of private browsing,
    // not silently ignoring it. https://bugs.webkit.org/show_bug.cgi?id=25894
    if (area->m_storageSyncManager) {
        area->m_storageAreaSync = StorageAreaSync::create(area->m_storageSyncManager.get(), area.copyRef(), area->m_securityOrigin->data().databaseIdentifier());
        ASSERT(area->m_storageAreaSync);
    }
    return area;
}

Ref<StorageAreaImpl> StorageAreaImpl::copy()
{
    ASSERT(!m_isShutdown);
    return adoptRef(*new StorageAreaImpl(*this));
}

StorageAreaImpl::StorageAreaImpl(const StorageAreaImpl& area)
    : m_storageType(area.m_storageType)
    , m_securityOrigin(area.m_securityOrigin)
    , m_storageMap(area.m_storageMap)
    , m_storageSyncManager(area.m_storageSyncManager)
#if ASSERT_ENABLED
    , m_isShutdown(area.m_isShutdown)
#endif
    , m_accessCount(0)
    , m_closeDatabaseTimer(*this, &StorageAreaImpl::closeDatabaseTimerFired)
{
    ASSERT(isMainThread());
    ASSERT(!m_isShutdown);
}

StorageType StorageAreaImpl::storageType() const
{
    return m_storageType;
}

unsigned StorageAreaImpl::length()
{
    ASSERT(!m_isShutdown);
    blockUntilImportComplete();

    return m_storageMap.length();
}

String StorageAreaImpl::key(unsigned index)
{
    ASSERT(!m_isShutdown);
    blockUntilImportComplete();

    return m_storageMap.key(index);
}

String StorageAreaImpl::item(const String& key)
{
    ASSERT(!m_isShutdown);
    blockUntilImportComplete();

    return m_storageMap.getItem(key);
}

void StorageAreaImpl::setItem(LocalFrame& sourceFrame, const String& key, const String& value, bool& quotaException)
{
    ASSERT(!m_isShutdown);
    ASSERT(!value.isNull());
    blockUntilImportComplete();

    String oldValue;
    m_storageMap.setItem(key, value, oldValue, quotaException);
    if (quotaException)
        return;

    if (oldValue == value)
        return;

    if (m_storageAreaSync)
        m_storageAreaSync->scheduleItemForSync(key, value);

    dispatchStorageEvent(key, oldValue, value, sourceFrame);
}

void StorageAreaImpl::removeItem(LocalFrame& sourceFrame, const String& key)
{
    ASSERT(!m_isShutdown);
    blockUntilImportComplete();

    String oldValue;
    m_storageMap.removeItem(key, oldValue);
    if (oldValue.isNull())
        return;

    if (m_storageAreaSync)
        m_storageAreaSync->scheduleItemForSync(key, String());

    dispatchStorageEvent(key, oldValue, String(), sourceFrame);
}

void StorageAreaImpl::clear(LocalFrame& sourceFrame)
{
    ASSERT(!m_isShutdown);
    blockUntilImportComplete();

    if (!m_storageMap.length())
        return;

    m_storageMap.clear();

    if (m_storageAreaSync)
        m_storageAreaSync->scheduleClear();

    dispatchStorageEvent(String(), String(), String(), sourceFrame);
}

bool StorageAreaImpl::contains(const String& key)
{
    ASSERT(!m_isShutdown);
    blockUntilImportComplete();

    return m_storageMap.contains(key);
}

void StorageAreaImpl::importItems(HashMap<String, String>&& items)
{
    ASSERT(!m_isShutdown);
    ASSERT(!isMainThread());

    m_storageMap.importItems(WTFMove(items));
}

void StorageAreaImpl::close()
{
    if (m_storageAreaSync)
        m_storageAreaSync->scheduleFinalSync();

#if ASSERT_ENABLED
    m_isShutdown = true;
#endif
}

void StorageAreaImpl::clearForOriginDeletion()
{
    ASSERT(!m_isShutdown);
    blockUntilImportComplete();

    m_storageMap.clear();

    if (m_storageAreaSync) {
        m_storageAreaSync->scheduleClear();
        m_storageAreaSync->scheduleCloseDatabase();
    }
}

void StorageAreaImpl::sync()
{
    ASSERT(!m_isShutdown);
    blockUntilImportComplete();

    if (m_storageAreaSync)
        m_storageAreaSync->scheduleSync();
}

void StorageAreaImpl::blockUntilImportComplete() const
{
    if (m_storageAreaSync)
        m_storageAreaSync->blockUntilImportComplete();
}

size_t StorageAreaImpl::memoryBytesUsedByCache()
{
    return 0;
}

void StorageAreaImpl::incrementAccessCount()
{
    m_accessCount++;

    if (m_closeDatabaseTimer.isActive())
        m_closeDatabaseTimer.stop();
}

void StorageAreaImpl::decrementAccessCount()
{
    ASSERT(m_accessCount);
    --m_accessCount;

    if (!m_accessCount) {
        if (m_closeDatabaseTimer.isActive())
            m_closeDatabaseTimer.stop();
        m_closeDatabaseTimer.startOneShot(StorageTracker::tracker().storageDatabaseIdleInterval());
    }
}

void StorageAreaImpl::closeDatabaseTimerFired()
{
    blockUntilImportComplete();
    if (m_storageAreaSync)
        m_storageAreaSync->scheduleCloseDatabase();
}

void StorageAreaImpl::closeDatabaseIfIdle()
{
    if (m_closeDatabaseTimer.isActive()) {
        ASSERT(!m_accessCount);
        m_closeDatabaseTimer.stop();

        closeDatabaseTimerFired();
    }
}

void StorageAreaImpl::dispatchStorageEvent(const String& key, const String& oldValue, const String& newValue, LocalFrame& sourceFrame)
{
    auto* page = sourceFrame.page();
    if (!page)
        return;

    auto isSourceStorage = [&sourceFrame](Storage& storage) {
        return storage.frame() == &sourceFrame;
    };
    if (isLocalStorage(m_storageType))
        StorageEventDispatcher::dispatchLocalStorageEvents(key, oldValue, newValue, &page->group(), m_securityOrigin, sourceFrame.document()->url().string(), WTFMove(isSourceStorage));
    else
        StorageEventDispatcher::dispatchSessionStorageEvents(key, oldValue, newValue, *page, m_securityOrigin, sourceFrame.document()->url().string(), WTFMove(isSourceStorage));
}

void StorageAreaImpl::sessionChanged(bool isNewSessionPersistent)
{
    ASSERT(isMainThread());

    // If import is not completed, background storage thread may be modifying m_storageMap.
    blockUntilImportComplete();

    m_storageMap.clear();

    if (isNewSessionPersistent && !m_storageAreaSync && m_storageSyncManager) {
        m_storageAreaSync = StorageAreaSync::create(m_storageSyncManager.get(), *this, m_securityOrigin->data().databaseIdentifier());
        return;
    }

    if (!isNewSessionPersistent && m_storageAreaSync) {
        m_storageAreaSync->scheduleFinalSync();
        m_storageAreaSync = nullptr;
    }
}

} // namespace WebCore
