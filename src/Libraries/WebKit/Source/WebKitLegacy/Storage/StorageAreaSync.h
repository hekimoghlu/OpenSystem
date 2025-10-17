/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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

#include <WebCore/SQLiteDatabase.h>
#include <WebCore/Timer.h>
#include <wtf/Condition.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/text/StringHash.h>

namespace WebCore {
class StorageSyncManager;
}

namespace WebKit {

class StorageAreaImpl;

class StorageAreaSync : public ThreadSafeRefCounted<StorageAreaSync, WTF::DestructionThread::Main> {
public:
    static Ref<StorageAreaSync> create(RefPtr<WebCore::StorageSyncManager>&&, Ref<StorageAreaImpl>&&, const String& databaseIdentifier);
    ~StorageAreaSync();

    void scheduleFinalSync();
    void blockUntilImportComplete();

    void scheduleItemForSync(const String& key, const String& value);
    void scheduleClear();
    void scheduleCloseDatabase();

    void scheduleSync();

private:
    StorageAreaSync(RefPtr<WebCore::StorageSyncManager>&&, Ref<StorageAreaImpl>&&, const String& databaseIdentifier);

    WebCore::Timer m_syncTimer;
    HashMap<String, String> m_changedItems;
    bool m_itemsCleared;

    bool m_finalSyncScheduled;

    RefPtr<StorageAreaImpl> m_storageArea;
    RefPtr<WebCore::StorageSyncManager> m_syncManager;

    // The database handle will only ever be opened and used on the background thread.
    WebCore::SQLiteDatabase m_database;

    // The following members are subject to thread synchronization issues.
public:
    // Called from the background thread
    void performImport();
    void performSync();
    void deleteEmptyDatabase();

private:
    enum OpenDatabaseParamType {
        CreateIfNonExistent,
        SkipIfNonExistent
    };

    void syncTimerFired();
    void openDatabase(OpenDatabaseParamType openingStrategy);
    void sync(bool clearItems, const HashMap<String, String>& items);

    const String m_databaseIdentifier;

    Lock m_syncLock;
    HashMap<String, String> m_itemsPendingSync;
    bool m_clearItemsWhileSyncing;
    bool m_syncScheduled;
    bool m_syncInProgress;
    bool m_databaseOpenFailed;

    bool m_syncCloseDatabase;

    mutable Lock m_importLock;
    Condition m_importCondition;
    bool m_importComplete WTF_GUARDED_BY_LOCK(m_importLock);
    void markImported();
    void migrateItemTableIfNeeded();
};

} // namespace WebCore
