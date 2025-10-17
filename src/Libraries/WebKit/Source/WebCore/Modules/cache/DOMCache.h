/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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

#include "ActiveDOMObject.h"
#include "CacheStorageConnection.h"
#include "FetchRequest.h"
#include "FetchResponse.h"
#include <wtf/UniqueRef.h>

namespace WebCore {

class ScriptExecutionContext;

class DOMCache final : public RefCounted<DOMCache>, public ActiveDOMObject {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<DOMCache> create(ScriptExecutionContext&, String&&, DOMCacheIdentifier, Ref<CacheStorageConnection>&&);
    ~DOMCache();

    using RequestInfo = FetchRequest::Info;

    using KeysPromise = DOMPromiseDeferred<IDLSequence<IDLInterface<FetchRequest>>>;

    void match(RequestInfo&&, CacheQueryOptions&&, Ref<DeferredPromise>&&);

    using MatchAllPromise = DOMPromiseDeferred<IDLSequence<IDLInterface<FetchResponse>>>;
    void matchAll(std::optional<RequestInfo>&&, CacheQueryOptions&&, MatchAllPromise&&);
    void add(RequestInfo&&, DOMPromiseDeferred<void>&&);

    void addAll(Vector<RequestInfo>&&, DOMPromiseDeferred<void>&&);
    void put(RequestInfo&&, Ref<FetchResponse>&&, DOMPromiseDeferred<void>&&);
    void remove(RequestInfo&&, CacheQueryOptions&&, DOMPromiseDeferred<IDLBoolean>&&);
    void keys(std::optional<RequestInfo>&&, CacheQueryOptions&&, KeysPromise&&);

    const String& name() const { return m_name; }
    DOMCacheIdentifier identifier() const { return m_identifier; }

    using MatchCallback = CompletionHandler<void(ExceptionOr<RefPtr<FetchResponse>>)>;
    void doMatch(RequestInfo&&, CacheQueryOptions&&, MatchCallback&&);

    CacheStorageConnection& connection() { return m_connection.get(); }

private:
    DOMCache(ScriptExecutionContext&, String&& name, DOMCacheIdentifier, Ref<CacheStorageConnection>&&);

    ExceptionOr<Ref<FetchRequest>> requestFromInfo(RequestInfo&&, bool ignoreMethod, bool* requestValidationFailed = nullptr);

    // ActiveDOMObject
    void stop() final;

    void putWithResponseData(DOMPromiseDeferred<void>&&, Ref<FetchRequest>&&, Ref<FetchResponse>&&, ExceptionOr<RefPtr<SharedBuffer>>&&);

    enum class ShouldRetrieveResponses : bool { No, Yes };
    using RecordsCallback = CompletionHandler<void(ExceptionOr<Vector<DOMCacheEngine::Record>>&&)>;
    void queryCache(ResourceRequest&&, const CacheQueryOptions&, ShouldRetrieveResponses, RecordsCallback&&);

    void batchDeleteOperation(const FetchRequest&, CacheQueryOptions&&, CompletionHandler<void(ExceptionOr<bool>&&)>&&);
    void batchPutOperation(const FetchRequest&, FetchResponse&, DOMCacheEngine::ResponseBody&&, CompletionHandler<void(ExceptionOr<void>&&)>&&);
    void batchPutOperation(Vector<DOMCacheEngine::Record>&&, CompletionHandler<void(ExceptionOr<void>&&)>&&);

    Vector<Ref<FetchResponse>> cloneResponses(const Vector<DOMCacheEngine::Record>&, MonotonicTime);
    DOMCacheEngine::Record toConnectionRecord(const FetchRequest&, FetchResponse&, DOMCacheEngine::ResponseBody&&);

    String m_name;
    DOMCacheIdentifier m_identifier;
    Ref<CacheStorageConnection> m_connection;

    bool m_isStopped { false };
};

} // namespace WebCore
