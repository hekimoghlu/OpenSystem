/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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

#include "CacheStorageRecord.h"
#include "CacheStorageStore.h"
#include "NetworkCacheKey.h"
#include <WebCore/RetrieveRecordsOptions.h>
#include <wtf/Identified.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WorkQueue.h>

namespace WebKit {
class CacheStorageCache;
}

namespace WebKit {

class CacheStorageManager;

class CacheStorageCache : public RefCountedAndCanMakeWeakPtr<CacheStorageCache>, public Identified<WebCore::DOMCacheIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED(CacheStorageCache);
public:
    static Ref<CacheStorageCache> create(CacheStorageManager&, const String& name, const String& uniqueName, const String& path, Ref<WorkQueue>&&);

    ~CacheStorageCache();
    const String& name() const { return m_name; }
    const String& uniqueName() const { return m_uniqueName; }
    CacheStorageManager* manager();

    void getSize(CompletionHandler<void(uint64_t)>&&);
    void open(WebCore::DOMCacheEngine::CacheIdentifierCallback&&);
    void retrieveRecords(WebCore::RetrieveRecordsOptions&&, WebCore::DOMCacheEngine::CrossThreadRecordsCallback&&);
    void removeRecords(WebCore::ResourceRequest&&, WebCore::CacheQueryOptions&&, WebCore::DOMCacheEngine::RecordIdentifiersCallback&&);
    void putRecords(Vector<WebCore::DOMCacheEngine::CrossThreadRecord>&&, WebCore::DOMCacheEngine::RecordIdentifiersCallback&&);
    void removeAllRecords();
    void close();

private:
    CacheStorageCache(CacheStorageManager&, const String& name, const String& uniqueName, const String& path, Ref<WorkQueue>&&);
    CacheStorageRecordInformation* findExistingRecord(const WebCore::ResourceRequest&, std::optional<uint64_t> = std::nullopt);
    void putRecordsAfterQuotaCheck(Vector<CacheStorageRecord>&&, WebCore::DOMCacheEngine::RecordIdentifiersCallback&&);
    void putRecordsInStore(Vector<CacheStorageRecord>&&, Vector<std::optional<CacheStorageRecord>>&&, WebCore::DOMCacheEngine::RecordIdentifiersCallback&&);
    void assertIsOnCorrectQueue() const
    {
#if ASSERT_ENABLED
        assertIsCurrent(m_queue.get());
#endif
    }

    static String computeKeyURL(const URL&);
    using RecordsMap = HashMap<String, Vector<CacheStorageRecordInformation>>;

    WeakPtr<CacheStorageManager> m_manager;
    bool m_isInitialized { false };
    Vector<WebCore::DOMCacheEngine::CacheIdentifierCallback> m_pendingInitializationCallbacks;
    String m_name;
    String m_uniqueName;
    RecordsMap m_records;
#if ASSERT_ENABLED
    const Ref<WorkQueue> m_queue;
#endif
    const Ref<CacheStorageStore> m_store;
};

} // namespace WebKit
