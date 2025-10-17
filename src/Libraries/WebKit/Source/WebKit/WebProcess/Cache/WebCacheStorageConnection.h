/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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

#include <WebCore/CacheStorageConnection.h>
#include <WebCore/ClientOrigin.h>
#include <wtf/HashCountedSet.h>
#include <wtf/Lock.h>

namespace IPC {
class Connection;
class Decoder;
class Encoder;
}

namespace WebKit {

class WebCacheStorageProvider;

class WebCacheStorageConnection final : public WebCore::CacheStorageConnection {
public:
    static Ref<WebCacheStorageConnection> create() { return adoptRef(*new WebCacheStorageConnection); }

    ~WebCacheStorageConnection();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);
    void networkProcessConnectionClosed();

private:
    WebCacheStorageConnection();

    struct PromiseConverter;

    // WebCore::CacheStorageConnection
    Ref<OpenPromise> open(const WebCore::ClientOrigin&, const String& cacheName) final;
    Ref<RemovePromise> remove(WebCore::DOMCacheIdentifier) final;
    Ref<RetrieveCachesPromise> retrieveCaches(const WebCore::ClientOrigin&, uint64_t)  final;
    Ref<RetrieveRecordsPromise> retrieveRecords(WebCore::DOMCacheIdentifier, WebCore::RetrieveRecordsOptions&&)  final;
    Ref<BatchPromise> batchDeleteOperation(WebCore::DOMCacheIdentifier, const WebCore::ResourceRequest&, WebCore::CacheQueryOptions&&)  final;
    Ref<BatchPromise> batchPutOperation(WebCore::DOMCacheIdentifier, Vector<WebCore::DOMCacheEngine::CrossThreadRecord>&&)  final;
    void reference(WebCore::DOMCacheIdentifier) final;
    void dereference(WebCore::DOMCacheIdentifier) final;
    void lockStorage(const WebCore::ClientOrigin&) final;
    void unlockStorage(const WebCore::ClientOrigin&) final;
    Ref<CompletionPromise> clearMemoryRepresentation(const WebCore::ClientOrigin&) final;
    Ref<EngineRepresentationPromise> engineRepresentation() final;
    void updateQuotaBasedOnSpaceUsage(const WebCore::ClientOrigin&) final;

    Ref<IPC::Connection> connection();

    Lock m_connectionLock;
    RefPtr<IPC::Connection> m_connection WTF_GUARDED_BY_LOCK(m_connectionLock);
    HashCountedSet<WebCore::DOMCacheIdentifier> m_connectedIdentifierCounters WTF_GUARDED_BY_LOCK(m_connectionLock);
    HashCountedSet<WebCore::ClientOrigin> m_clientOriginLockRequestCounters WTF_GUARDED_BY_LOCK(m_connectionLock);
};

}
