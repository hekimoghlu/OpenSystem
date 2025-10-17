/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include <wtf/HashSet.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class StorageThread;
class SecurityOrigin;
class StorageTrackerClient;
class SecurityOriginData;
}

namespace WebKit {

class StorageTracker {
    WTF_MAKE_NONCOPYABLE(StorageTracker);
    WTF_MAKE_TZONE_ALLOCATED(StorageTracker);
public:
    static void initializeTracker(const String& storagePath, WebCore::StorageTrackerClient*);
    static StorageTracker& tracker();

    void setOriginDetails(const String& originIdentifier, const String& databaseFile);
    
    void deleteAllOrigins();
    void deleteOrigin(const WebCore::SecurityOriginData&);
    void deleteOriginWithIdentifier(const String& originIdentifier);
    Vector<WebCore::SecurityOriginData> origins();
    uint64_t diskUsageForOrigin(WebCore::SecurityOrigin*);
    
    void cancelDeletingOrigin(const String& originIdentifier);
    
    bool isActive();

    Seconds storageDatabaseIdleInterval() { return m_StorageDatabaseIdleInterval; }
    void setStorageDatabaseIdleInterval(Seconds interval) { m_StorageDatabaseIdleInterval = interval; }

    void syncFileSystemAndTrackerDatabase();

private:
    explicit StorageTracker(const String& storagePath);

    void internalInitialize();

    String trackerDatabasePath();
    void openTrackerDatabase(bool createIfDoesNotExist);

    void importOriginIdentifiers();
    void finishedImportingOriginIdentifiers();
    
    void deleteTrackerFiles();
    String databasePathForOrigin(const String& originIdentifier);

    bool canDeleteOrigin(const String& originIdentifier);
    void willDeleteOrigin(const String& originIdentifier);
    void willDeleteAllOrigins();

    void originFilePaths(Vector<String>& paths);
    
    void setIsActive(bool);

    // Sync to disk on background thread.
    void syncDeleteAllOrigins();
    void syncDeleteOrigin(const String& originIdentifier);
    void syncSetOriginDetails(const String& originIdentifier, const String& databaseFile);
    void syncImportOriginIdentifiers();

    // Mutex for m_database and m_storageDirectoryPath.
    Lock m_databaseMutex;
    WebCore::SQLiteDatabase m_database;
    String m_storageDirectoryPath;

    Lock m_clientMutex;
    WebCore::StorageTrackerClient* m_client;

    // Guard for m_originSet and m_originsBeingDeleted.
    Lock m_originSetMutex;
    typedef HashSet<String> OriginSet;
    OriginSet m_originSet;
    OriginSet m_originsBeingDeleted;

    std::unique_ptr<WebCore::StorageThread> m_thread;
    
    bool m_isActive;
    bool m_needsInitialization;
    Seconds m_StorageDatabaseIdleInterval;
};
    
} // namespace WebCore
