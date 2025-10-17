/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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

#include <WebCore/UniqueIDBDatabase.h>
#include <WebCore/UniqueIDBDatabaseManager.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class IDBRequestData;

namespace IDBServer {
class IDBBackingStore;
class IDBConnectionToClient;
class UniqueIDBTransaction;
}

struct IDBDatabaseNameAndVersion;
}

namespace WebKit {

class IDBStorageRegistry;

class IDBStorageManager final : public WebCore::IDBServer::UniqueIDBDatabaseManager {
    WTF_MAKE_TZONE_ALLOCATED(IDBStorageManager);
public:
    static void createVersionDirectoryIfNeeded(const String& rootDirectory);
    static String idbStorageOriginDirectory(const String& rootDirectory, const WebCore::ClientOrigin&);
    static uint64_t idbStorageSize(const String& originDirectory);
    static HashSet<WebCore::ClientOrigin> originsOfIDBStorageData(const String& rootDirectory);
    static bool migrateOriginData(const String& oldOriginDirectory, const String& newOriginDirectory);

    using QuotaCheckFunction = Function<void(uint64_t spaceRequested, CompletionHandler<void(bool)>&&)>;
    IDBStorageManager(const String& path, IDBStorageRegistry&, QuotaCheckFunction&&);
    ~IDBStorageManager();
    bool isActive() const;
    bool hasDataInMemory() const;
    void closeDatabasesForDeletion();
    void stopDatabaseActivitiesForSuspend();
    void handleLowMemoryWarning();

    void openDatabase(WebCore::IDBServer::IDBConnectionToClient&, const WebCore::IDBOpenRequestData&);
    void openDBRequestCancelled(const WebCore::IDBOpenRequestData&);
    void deleteDatabase(WebCore::IDBServer::IDBConnectionToClient&, const WebCore::IDBOpenRequestData&);
    Vector<WebCore::IDBDatabaseNameAndVersion> getAllDatabaseNamesAndVersions();

private:
    WebCore::IDBServer::UniqueIDBDatabase& getOrCreateUniqueIDBDatabase(const WebCore::IDBDatabaseIdentifier&);

    // WebCore::UniqueIDBDatabaseManager
    void registerConnection(WebCore::IDBServer::UniqueIDBDatabaseConnection&) final;
    void unregisterConnection(WebCore::IDBServer::UniqueIDBDatabaseConnection&) final;
    void registerTransaction(WebCore::IDBServer::UniqueIDBDatabaseTransaction&) final;
    void unregisterTransaction(WebCore::IDBServer::UniqueIDBDatabaseTransaction&) final;
    std::unique_ptr<WebCore::IDBServer::IDBBackingStore> createBackingStore(const WebCore::IDBDatabaseIdentifier&) final;
    void requestSpace(const WebCore::ClientOrigin&, uint64_t size, CompletionHandler<void(bool)>&&) final;

    String m_path;
    CheckedRef<IDBStorageRegistry> m_registry;
    QuotaCheckFunction m_quotaCheckFunction;
    HashMap<WebCore::IDBDatabaseIdentifier, std::unique_ptr<WebCore::IDBServer::UniqueIDBDatabase>> m_databases;
};

} // namespace WebKit
