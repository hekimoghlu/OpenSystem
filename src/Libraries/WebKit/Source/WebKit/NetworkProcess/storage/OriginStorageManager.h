/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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

#include "Connection.h"
#include "OriginQuotaManager.h"
#include "WebsiteDataType.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
class OriginStorageManager;
}

namespace WebCore {
struct ClientOrigin;
struct StorageEstimate;
}

namespace WebKit {

class BackgroundFetchStoreManager;
class CacheStorageManager;
class CacheStorageRegistry;
class FileSystemStorageHandleRegistry;
class FileSystemStorageManager;
class IDBStorageManager;
class IDBStorageRegistry;
class LocalStorageManager;
class ServiceWorkerStorageManager;
class SessionStorageManager;
class StorageAreaRegistry;

enum class UnifiedOriginStorageLevel : uint8_t;
enum class WebsiteDataType : uint32_t;

class OriginStorageManager final : public CanMakeWeakPtr<OriginStorageManager>, public CanMakeThreadSafeCheckedPtr<OriginStorageManager> {
    WTF_MAKE_TZONE_ALLOCATED(OriginStorageManager);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(OriginStorageManager);
public:
    static String originFileIdentifier();

    OriginStorageManager(OriginQuotaManager::Parameters&&, String&& path, String&& cusotmLocalStoragePath, String&& customIDBStoragePath, String&& customCacheStoragePath, UnifiedOriginStorageLevel);
    ~OriginStorageManager();

    void connectionClosed(IPC::Connection::UniqueID);
    WebCore::StorageEstimate estimate();
    const String& path() const { return m_path; }
    OriginQuotaManager& quotaManager();
    Ref<OriginQuotaManager> protectedQuotaManager();
    FileSystemStorageManager& fileSystemStorageManager(FileSystemStorageHandleRegistry&);
    FileSystemStorageManager* existingFileSystemStorageManager();
    LocalStorageManager& localStorageManager(StorageAreaRegistry&);
    LocalStorageManager* existingLocalStorageManager();
    SessionStorageManager& sessionStorageManager(StorageAreaRegistry&);
    SessionStorageManager* existingSessionStorageManager();
    IDBStorageManager& idbStorageManager(IDBStorageRegistry&);
    IDBStorageManager* existingIDBStorageManager();
    CacheStorageManager& cacheStorageManager(CacheStorageRegistry&, const WebCore::ClientOrigin&, Ref<WorkQueue>&&);
    Ref<CacheStorageManager> protectedCacheStorageManager(CacheStorageRegistry&, const WebCore::ClientOrigin&, Ref<WorkQueue>&&);
    CacheStorageManager* existingCacheStorageManager();
    BackgroundFetchStoreManager& backgroundFetchManager(Ref<WTF::WorkQueue>&&);
    ServiceWorkerStorageManager& serviceWorkerStorageManager();
    uint64_t cacheStorageSize();
    void closeCacheStorageManager();
    String resolvedPath(WebsiteDataType);
    bool isActive();
    bool hasDataInMemory();
    bool isEmpty();
    using DataTypeSizeMap = HashMap<WebsiteDataType, uint64_t, IntHash<WebsiteDataType>, WTF::StrongEnumHashTraits<WebsiteDataType>>;
    DataTypeSizeMap fetchDataTypesInList(OptionSet<WebsiteDataType>, bool shouldComputeSize);
    void deleteData(OptionSet<WebsiteDataType>, WallTime);
    void moveData(OptionSet<WebsiteDataType>, const String& localStoragePath, const String& idbStoragePath);
    void deleteEmptyDirectory();
    std::optional<WallTime> originFileCreationTimestamp() const { return m_originFileCreationTimestamp; }
    void setOriginFileCreationTimestamp(std::optional<WallTime> timestamp) { m_originFileCreationTimestamp = timestamp; }
#if PLATFORM(IOS_FAMILY)
    bool includedInBackup() const { return m_includedInBackup; }
    void markIncludedInBackup() { m_includedInBackup = true; }
#endif

private:
    Ref<OriginQuotaManager> createQuotaManager(OriginQuotaManager::Parameters&&);
    enum class StorageBucketMode : bool;
    class StorageBucket;
    StorageBucket& defaultBucket();

    std::unique_ptr<StorageBucket> m_defaultBucket;
    String m_path;
    String m_customLocalStoragePath;
    String m_customIDBStoragePath;
    String m_customCacheStoragePath;
    Ref<OriginQuotaManager> m_quotaManager;
    UnifiedOriginStorageLevel m_level;
    Markable<WallTime> m_originFileCreationTimestamp;
#if PLATFORM(IOS_FAMILY)
    bool m_includedInBackup { false };
#endif
};

} // namespace WebKit

