/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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

#include <WebCore/BackgroundFetchStore.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/Function.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WTF {
class WorkQueue;
}

namespace WebKit {

class BackgroundFetchStoreManager : public RefCountedAndCanMakeWeakPtr<BackgroundFetchStoreManager> {
    WTF_MAKE_TZONE_ALLOCATED(BackgroundFetchStoreManager);
public:
    using QuotaCheckFunction = Function<void(uint64_t spaceRequested, CompletionHandler<void(bool)>&&)>;

    static Ref<BackgroundFetchStoreManager> create(const String& path, Ref<WTF::WorkQueue>&& taskQueue, QuotaCheckFunction&& quotaCheckFunction)
    {
        return adoptRef(*new BackgroundFetchStoreManager(path, WTFMove(taskQueue), WTFMove(quotaCheckFunction)));
    }
    ~BackgroundFetchStoreManager();

    static String createNewStorageIdentifier();

    using InitializeCallback = CompletionHandler<void(Vector<std::pair<RefPtr<WebCore::SharedBuffer>, String>>&&)>;
    void initializeFetches(InitializeCallback&&);

    void clearFetch(const String&);
    void clearFetch(const String&, CompletionHandler<void()>&&);
    void clearAllFetches(const Vector<String>&, CompletionHandler<void()>&&);

    using StoreResult = WebCore::BackgroundFetchStore::StoreResult;
    void storeFetch(const String&, uint64_t downloadTotal, uint64_t uploadTotal, std::optional<size_t> responseBodyIndexToClear, Vector<uint8_t>&&, CompletionHandler<void(StoreResult)>&&);
    void storeFetchResponseBodyChunk(const String&, size_t, const WebCore::SharedBuffer&, CompletionHandler<void(StoreResult)>&&);

    void retrieveResponseBody(const String&, size_t, CompletionHandler<void(RefPtr<WebCore::SharedBuffer>&&)>&&);

private:
    BackgroundFetchStoreManager(const String&, Ref<WTF::WorkQueue>&&, QuotaCheckFunction&&);

    void storeFetchAfterQuotaCheck(const String&, uint64_t downloadTotal, uint64_t uploadTotal, std::optional<size_t> responseBodyIndexToClear, Vector<uint8_t>&&, CompletionHandler<void(StoreResult)>&&);

    String m_path;
    Ref<WTF::WorkQueue> m_taskQueue;
    Ref<WTF::WorkQueue> m_ioQueue;
    QuotaCheckFunction m_quotaCheckFunction;

    HashMap<String, Vector<WebCore::SharedBufferBuilder>> m_nonPersistentChunks;
};

} // namespace WebKit
