/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include <WebCore/DOMCacheEngine.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {
class CacheStorageManager;
}

namespace WebCore {
struct ClientOrigin;
}

namespace WebKit {
class CacheStorageCache;
class CacheStorageRecordInformation;
class CacheStorageRegistry;
class CacheStorageStore;
struct CacheStorageRecord;


class CacheStorageManager : public RefCountedAndCanMakeWeakPtr<CacheStorageManager> {
    WTF_MAKE_TZONE_ALLOCATED(CacheStorageManager);
public:
    static String cacheStorageOriginDirectory(const String& rootDirectory, const WebCore::ClientOrigin&);
    static void copySaltFileToOriginDirectory(const String& rootDirectory, const String& originDirectory);
    static HashSet<WebCore::ClientOrigin> originsOfCacheStorageData(const String& rootDirectory);
    static uint64_t cacheStorageSize(const String& originDirectory);
    static bool hasCacheList(const String& cacheListDirectory);

    using QuotaCheckFunction = Function<void(uint64_t spaceRequested, CompletionHandler<void(bool)>&&)>;
    static Ref<CacheStorageManager> create(const String& path, CacheStorageRegistry&, const std::optional<WebCore::ClientOrigin>&, QuotaCheckFunction&&, Ref<WorkQueue>&&);
    ~CacheStorageManager();
    void openCache(const String& name, WebCore::DOMCacheEngine::CacheIdentifierCallback&&);
    void removeCache(WebCore::DOMCacheIdentifier, WebCore::DOMCacheEngine::RemoveCacheIdentifierCallback&&);
    void allCaches(uint64_t updateCounter, WebCore::DOMCacheEngine::CacheInfosCallback&&);
    void reference(IPC::Connection::UniqueID, WebCore::DOMCacheIdentifier);
    void dereference(IPC::Connection::UniqueID, WebCore::DOMCacheIdentifier);
    void lockStorage(IPC::Connection::UniqueID);
    void unlockStorage(IPC::Connection::UniqueID);

    void connectionClosed(IPC::Connection::UniqueID);
    bool hasDataInMemory();
    bool isActive();
    String representationString();
    FileSystem::Salt salt() const { return m_salt; }
    void requestSpace(uint64_t size, CompletionHandler<void(bool)>&&);
    void sizeIncreased(uint64_t amount);
    void sizeDecreased(uint64_t amount);
    void reset();

private:
    CacheStorageManager(const String& path, CacheStorageRegistry&, const std::optional<WebCore::ClientOrigin>&, QuotaCheckFunction&&, Ref<WorkQueue>&&);
    void makeDirty();
    bool initializeCaches();
    void removeUnusedCache(WebCore::DOMCacheIdentifier);
    void initializeCacheSize(CacheStorageCache&);
    void finishInitializingSize();
    void requestSpaceAfterInitializingSize(uint64_t size, CompletionHandler<void(bool)>&&);
    Ref<CacheStorageRegistry> protectedRegistry();

    bool m_isInitialized { false };
    uint64_t m_updateCounter;
    std::optional<uint64_t> m_size;
    std::pair<uint64_t, HashSet<WebCore::DOMCacheIdentifier>> m_pendingSize;
    String m_path;
    FileSystem::Salt m_salt;
    Ref<CacheStorageRegistry> m_registry;
    QuotaCheckFunction m_quotaCheckFunction;
    Vector<Ref<CacheStorageCache>> m_caches;
    HashMap<WebCore::DOMCacheIdentifier, Ref<CacheStorageCache>> m_removedCaches;
    HashMap<WebCore::DOMCacheIdentifier, Vector<IPC::Connection::UniqueID>> m_cacheRefConnections;
    HashSet<IPC::Connection::UniqueID> m_activeConnections;
    Ref<WorkQueue> m_queue;
    Deque<std::pair<uint64_t, CompletionHandler<void(bool)>>> m_pendingSpaceRequests;
};

} // namespace WebKit
