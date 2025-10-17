/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

#include "CacheStorageConnection.h"
#include <wtf/NativePromise.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class CacheStorageProvider : public RefCounted<CacheStorageProvider> {
public:
    class DummyCacheStorageConnection final : public WebCore::CacheStorageConnection {
    public:
        static Ref<DummyCacheStorageConnection> create() { return adoptRef(*new DummyCacheStorageConnection()); }

    private:
        DummyCacheStorageConnection()
        {
        }

        Ref<OpenPromise> open(const ClientOrigin&, const String&) final { return OpenPromise::createAndReject(DOMCacheEngine::Error::Stopped); }
        Ref<RemovePromise> remove(DOMCacheIdentifier) final { return RemovePromise::createAndReject(DOMCacheEngine::Error::Stopped); }
        Ref<RetrieveCachesPromise> retrieveCaches(const ClientOrigin&, uint64_t)  final { return RetrieveCachesPromise::createAndReject(DOMCacheEngine::Error::Stopped); }
        Ref<RetrieveRecordsPromise> retrieveRecords(DOMCacheIdentifier, RetrieveRecordsOptions&&)  final { return RetrieveRecordsPromise::createAndReject(DOMCacheEngine::Error::Stopped); }
        Ref<BatchPromise> batchDeleteOperation(DOMCacheIdentifier, const ResourceRequest&, CacheQueryOptions&&)  final { return BatchPromise::createAndReject(DOMCacheEngine::Error::Stopped); }
        Ref<BatchPromise> batchPutOperation(DOMCacheIdentifier, Vector<DOMCacheEngine::CrossThreadRecord>&&)  final { return BatchPromise::createAndReject(DOMCacheEngine::Error::Stopped); }
        void reference(DOMCacheIdentifier) final { }
        void dereference(DOMCacheIdentifier) final { }
        void lockStorage(const ClientOrigin&) final { }
        void unlockStorage(const ClientOrigin&) final { }
    };

    static Ref<CacheStorageProvider> create() { return adoptRef(*new CacheStorageProvider); }
    virtual Ref<CacheStorageConnection> createCacheStorageConnection() { return DummyCacheStorageConnection::create(); }
    virtual ~CacheStorageProvider() { };

protected:
    CacheStorageProvider() = default;
};

}
