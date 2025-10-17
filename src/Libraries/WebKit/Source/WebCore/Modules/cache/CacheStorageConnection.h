/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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

#include "DOMCacheEngine.h"
#include "RetrieveRecordsOptions.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/NativePromise.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

struct ClientOrigin;
class FetchResponse;

class CacheStorageConnection : public ThreadSafeRefCounted<CacheStorageConnection> {
public:
    virtual ~CacheStorageConnection() = default;

    using OpenPromise = NativePromise<DOMCacheEngine::CacheIdentifierOperationResult, DOMCacheEngine::Error>;
    virtual Ref<OpenPromise> open(const ClientOrigin&, const String& cacheName) = 0;
    using RemovePromise = NativePromise<bool, DOMCacheEngine::Error>;
    virtual Ref<RemovePromise> remove(DOMCacheIdentifier) = 0;
    using RetrieveCachesPromise = NativePromise<DOMCacheEngine::CacheInfos, DOMCacheEngine::Error>;
    virtual Ref<RetrieveCachesPromise> retrieveCaches(const ClientOrigin&, uint64_t updateCounter) = 0;

    using RetrieveRecordsPromise = NativePromise<Vector<DOMCacheEngine::CrossThreadRecord>, DOMCacheEngine::Error>;
    virtual Ref<RetrieveRecordsPromise> retrieveRecords(DOMCacheIdentifier, RetrieveRecordsOptions&&) = 0;
    using BatchPromise = NativePromise<Vector<uint64_t>, DOMCacheEngine::Error>;
    virtual Ref<BatchPromise> batchDeleteOperation(DOMCacheIdentifier, const ResourceRequest&, CacheQueryOptions&&) = 0;
    virtual Ref<BatchPromise> batchPutOperation(DOMCacheIdentifier, Vector<DOMCacheEngine::CrossThreadRecord>&&) = 0;

    virtual void reference(DOMCacheIdentifier /* cacheIdentifier */) = 0;
    virtual void dereference(DOMCacheIdentifier /* cacheIdentifier */) = 0;
    virtual void lockStorage(const ClientOrigin&) = 0;
    virtual void unlockStorage(const ClientOrigin&) = 0;

    uint64_t computeRecordBodySize(const FetchResponse&, const DOMCacheEngine::ResponseBody&);

    // Used only for testing purposes.
    using CompletionPromise = NativePromise<void, DOMCacheEngine::Error>;
    virtual Ref<CompletionPromise> clearMemoryRepresentation(const ClientOrigin&) { return CompletionPromise::createAndReject(DOMCacheEngine::Error::NotImplemented); }
    using EngineRepresentationPromise = NativePromise<String, DOMCacheEngine::Error>;
    virtual Ref<EngineRepresentationPromise> engineRepresentation() { return EngineRepresentationPromise::createAndReject(DOMCacheEngine::Error::NotImplemented); }
    virtual void updateQuotaBasedOnSpaceUsage(const ClientOrigin&) { }

private:
    uint64_t computeRealBodySize(const DOMCacheEngine::ResponseBody&);

protected:
    HashMap<uint64_t, uint64_t> m_opaqueResponseToSizeWithPaddingMap;
};

} // namespace WebCore
