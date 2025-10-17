/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#include "DOMCache.h"
#include "FetchRequest.h"
#include <wtf/Forward.h>

namespace WebCore {

struct MultiCacheQueryOptions;

class DOMCacheStorage : public RefCounted<DOMCacheStorage>, public ActiveDOMObject {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<DOMCacheStorage> create(ScriptExecutionContext&, Ref<CacheStorageConnection>&&);
    ~DOMCacheStorage();

    using KeysPromise = DOMPromiseDeferred<IDLSequence<IDLDOMString>>;

    void match(DOMCache::RequestInfo&&, MultiCacheQueryOptions&&, Ref<DeferredPromise>&&);
    void has(const String&, DOMPromiseDeferred<IDLBoolean>&&);
    void open(const String&, DOMPromiseDeferred<IDLInterface<DOMCache>>&&);
    void remove(const String&, DOMPromiseDeferred<IDLBoolean>&&);
    void keys(KeysPromise&&);

private:
    DOMCacheStorage(ScriptExecutionContext&, Ref<CacheStorageConnection>&&);

    // ActiveDOMObject
    void stop() final;

    void doOpen(const String& name, DOMPromiseDeferred<IDLInterface<DOMCache>>&&);
    void doRemove(const String&, DOMPromiseDeferred<IDLBoolean>&&);
    void doSequentialMatch(DOMCache::RequestInfo&&, CacheQueryOptions&&, Ref<DeferredPromise>&&);
    void retrieveCaches(CompletionHandler<void(std::optional<Exception>&&)>&&);
    Ref<DOMCache> findCacheOrCreate(DOMCacheEngine::CacheInfo&&, ScriptExecutionContext&);
    std::optional<ClientOrigin> origin() const;

    Vector<Ref<DOMCache>> m_caches;
    uint64_t m_updateCounter { 0 };
    Ref<CacheStorageConnection> m_connection;
    bool m_isStopped { false };
};

} // namespace WebCore
