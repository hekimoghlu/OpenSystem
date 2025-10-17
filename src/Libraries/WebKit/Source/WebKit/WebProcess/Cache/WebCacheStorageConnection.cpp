/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#include "config.h"
#include "WebCacheStorageConnection.h"

#include "NetworkConnectionToWebProcessMessages.h"
#include "NetworkProcessConnection.h"
#include "NetworkProcessMessages.h"
#include "NetworkStorageManagerMessages.h"
#include "WebCacheStorageProvider.h"
#include "WebProcess.h"
#include <wtf/MainThread.h>
#include <wtf/NativePromise.h>

namespace WebKit {

WebCacheStorageConnection::WebCacheStorageConnection()
{
}

WebCacheStorageConnection::~WebCacheStorageConnection()
{
}

Ref<IPC::Connection> WebCacheStorageConnection::connection()
{
    {
        Locker lock(m_connectionLock);
        if (m_connection)
            return *m_connection;
    }

    RefPtr<IPC::Connection> connection;
    callOnMainRunLoopAndWait([this, &connection]() mutable {
        connection = &WebProcess::singleton().ensureNetworkProcessConnection().connection();
        {
            Locker lock(m_connectionLock);
            m_connection = connection;
        }
    });

    return connection.releaseNonNull();
}

struct WebCacheStorageConnection::PromiseConverter {
    static auto convertError(IPC::Error)
    {
        return makeUnexpected(WebCore::DOMCacheEngine::Error::Internal);
    }
};

auto WebCacheStorageConnection::open(const WebCore::ClientOrigin& origin, const String& cacheName) -> Ref<OpenPromise>
{
    return connection()->sendWithPromisedReply<PromiseConverter>(Messages::NetworkStorageManager::CacheStorageOpenCache { origin, cacheName });
}

auto WebCacheStorageConnection::remove(WebCore::DOMCacheIdentifier cacheIdentifier) -> Ref<RemovePromise>
{
    return connection()->sendWithPromisedReply<PromiseConverter>(Messages::NetworkStorageManager::CacheStorageRemoveCache { cacheIdentifier });
}

auto WebCacheStorageConnection::retrieveCaches(const WebCore::ClientOrigin& origin, uint64_t updateCounter) -> Ref<RetrieveCachesPromise>
{
    return connection()->sendWithPromisedReply<PromiseConverter>(Messages::NetworkStorageManager::CacheStorageAllCaches { origin, updateCounter });
}

auto WebCacheStorageConnection::retrieveRecords(WebCore::DOMCacheIdentifier cacheIdentifier, WebCore::RetrieveRecordsOptions&& options) -> Ref<RetrieveRecordsPromise>
{
    return connection()->sendWithPromisedReply<PromiseConverter>(Messages::NetworkStorageManager::CacheStorageRetrieveRecords { cacheIdentifier, options });
}

auto WebCacheStorageConnection::batchDeleteOperation(WebCore::DOMCacheIdentifier cacheIdentifier, const WebCore::ResourceRequest& request, WebCore::CacheQueryOptions&& options) -> Ref<BatchPromise>
{
    return connection()->sendWithPromisedReply<PromiseConverter>(Messages::NetworkStorageManager::CacheStorageRemoveRecords { cacheIdentifier, request, options });
}

auto WebCacheStorageConnection::batchPutOperation(WebCore::DOMCacheIdentifier cacheIdentifier, Vector<WebCore::DOMCacheEngine::CrossThreadRecord>&& records) -> Ref<BatchPromise>
{
    return connection()->sendWithPromisedReply<PromiseConverter>(Messages::NetworkStorageManager::CacheStoragePutRecords { cacheIdentifier, WTFMove(records) });
}

void WebCacheStorageConnection::reference(WebCore::DOMCacheIdentifier cacheIdentifier)
{
    Locker connectionLocker { m_connectionLock };
    if (m_connectedIdentifierCounters.add(cacheIdentifier).isNewEntry && m_connection)
        m_connection->send(Messages::NetworkStorageManager::CacheStorageReference(cacheIdentifier), 0);
}

void WebCacheStorageConnection::dereference(WebCore::DOMCacheIdentifier cacheIdentifier)
{
    Locker connectionLocker { m_connectionLock };
    if (m_connectedIdentifierCounters.remove(cacheIdentifier) && m_connection)
        m_connection->send(Messages::NetworkStorageManager::CacheStorageDereference(cacheIdentifier), 0);
}

void WebCacheStorageConnection::lockStorage(const WebCore::ClientOrigin& origin)
{
    Locker connectionLocker { m_connectionLock };
    if (m_clientOriginLockRequestCounters.add(origin).isNewEntry && m_connection)
        m_connection->send(Messages::NetworkStorageManager::LockCacheStorage { origin }, 0);
}

void WebCacheStorageConnection::unlockStorage(const WebCore::ClientOrigin& origin)
{
    Locker connectionLocker { m_connectionLock };
    if (m_clientOriginLockRequestCounters.remove(origin) && m_connection)
        m_connection->send(Messages::NetworkStorageManager::UnlockCacheStorage { origin }, 0);
}

auto WebCacheStorageConnection::clearMemoryRepresentation(const WebCore::ClientOrigin& origin) -> Ref<CompletionPromise>
{
    return connection()->sendWithPromisedReply<PromiseConverter>(Messages::NetworkStorageManager::CacheStorageClearMemoryRepresentation { origin });
}

auto WebCacheStorageConnection::engineRepresentation() -> Ref<EngineRepresentationPromise>
{
    return connection()->sendWithPromisedReply<PromiseConverter>(Messages::NetworkStorageManager::CacheStorageRepresentation { });
}

void WebCacheStorageConnection::updateQuotaBasedOnSpaceUsage(const WebCore::ClientOrigin& origin)
{
    connection()->send(Messages::NetworkStorageManager::ResetQuotaUpdatedBasedOnUsageForTesting(origin), 0);
}

void WebCacheStorageConnection::networkProcessConnectionClosed()
{
    Locker connectionLocker { m_connectionLock };

    m_connectedIdentifierCounters.clear();
    m_clientOriginLockRequestCounters.clear();
    m_connection = nullptr;
}

}
