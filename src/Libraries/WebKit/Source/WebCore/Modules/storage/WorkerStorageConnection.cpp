/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#include "WorkerStorageConnection.h"

#include "ClientOrigin.h"
#include "Document.h"
#include "StorageEstimate.h"
#include "WorkerFileSystemStorageConnection.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerThread.h"
#include <wtf/Scope.h>

namespace WebCore {

Ref<WorkerStorageConnection> WorkerStorageConnection::create(WorkerGlobalScope& scope)
{
    return adoptRef(*new WorkerStorageConnection(scope));
}

WorkerStorageConnection::WorkerStorageConnection(WorkerGlobalScope& scope)
    : m_scope(scope)
{
}

void WorkerStorageConnection::scopeClosed()
{
    auto getPersistedCallbacks = std::exchange(m_getPersistedCallbacks, { });
    for (auto& callback : getPersistedCallbacks.values())
        callback(false);

    auto getDirectoryCallbacks = std::exchange(m_getDirectoryCallbacks, { });
    for (auto& callback : getDirectoryCallbacks.values())
        callback(Exception { ExceptionCode::InvalidStateError });

    m_scope = nullptr;
}

void WorkerStorageConnection::getPersisted(ClientOrigin&& origin, StorageConnection::PersistCallback&& completionHandler)
{
    ASSERT(m_scope);

    auto* workerLoaderProxy = m_scope->thread().workerLoaderProxy();
    if (!workerLoaderProxy)
        return completionHandler(false);

    auto callbackIdentifier = ++m_lastCallbackIdentifier;
    m_getPersistedCallbacks.add(callbackIdentifier, WTFMove(completionHandler));

    workerLoaderProxy->postTaskToLoader([callbackIdentifier, contextIdentifier = m_scope->identifier(), origin = WTFMove(origin).isolatedCopy()](auto& context) mutable {
        ASSERT(isMainThread());

        auto& document = downcast<Document>(context);
        auto mainThreadConnection = document.storageConnection();
        auto mainThreadCallback = [callbackIdentifier, contextIdentifier](bool result) mutable {
            ScriptExecutionContext::postTaskTo(contextIdentifier, [callbackIdentifier, result] (auto& scope) mutable {
                downcast<WorkerGlobalScope>(scope).storageConnection().didGetPersisted(callbackIdentifier, result);
            });
        };
        if (!mainThreadConnection)
            return mainThreadCallback(false);

        mainThreadConnection->getPersisted(WTFMove(origin), WTFMove(mainThreadCallback));
    });
}

void WorkerStorageConnection::didGetPersisted(uint64_t callbackIdentifier, bool persisted)
{
    if (auto callback = m_getPersistedCallbacks.take(callbackIdentifier))
        callback(persisted);
}

void WorkerStorageConnection::getEstimate(ClientOrigin&& origin, StorageConnection::GetEstimateCallback&& completionHandler)
{
    ASSERT(m_scope);

    auto* workerLoaderProxy = m_scope->thread().workerLoaderProxy();
    if (!workerLoaderProxy)
        return completionHandler(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = ++m_lastCallbackIdentifier;
    m_getEstimateCallbacks.add(callbackIdentifier, WTFMove(completionHandler));

    workerLoaderProxy->postTaskToLoader([callbackIdentifier, contextIdentifier = m_scope->identifier(), origin = WTFMove(origin).isolatedCopy()](auto& context) mutable {
        ASSERT(isMainThread());

        auto& document = downcast<Document>(context);
        auto mainThreadConnection = document.storageConnection();
        auto mainThreadCallback = [callbackIdentifier, contextIdentifier](ExceptionOr<StorageEstimate>&& result) mutable {
            ScriptExecutionContext::postTaskTo(contextIdentifier, [callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                downcast<WorkerGlobalScope>(scope).storageConnection().didGetEstimate(callbackIdentifier, WTFMove(result));
            });
        };
        if (!mainThreadConnection)
            return mainThreadCallback(Exception { ExceptionCode::InvalidStateError });

        mainThreadConnection->getEstimate(WTFMove(origin), WTFMove(mainThreadCallback));
    });
}

void WorkerStorageConnection::didGetEstimate(uint64_t callbackIdentifier, ExceptionOr<StorageEstimate>&& result)
{
    if (auto callback = m_getEstimateCallbacks.take(callbackIdentifier))
        callback(WTFMove(result));
}

void WorkerStorageConnection::fileSystemGetDirectory(ClientOrigin&& origin, StorageConnection::GetDirectoryCallback&& completionHandler)
{
    ASSERT(m_scope);

    auto* workerLoaderProxy = m_scope->thread().workerLoaderProxy();
    if (!workerLoaderProxy)
        return completionHandler(Exception { ExceptionCode::InvalidStateError });
    
    auto callbackIdentifier = ++m_lastCallbackIdentifier;
    m_getDirectoryCallbacks.add(callbackIdentifier, WTFMove(completionHandler));

    workerLoaderProxy->postTaskToLoader([callbackIdentifier, contextIdentifier = m_scope->identifier(), origin = WTFMove(origin).isolatedCopy()](auto& context) mutable {
        ASSERT(isMainThread());

        auto& document = downcast<Document>(context);
        auto mainThreadConnection = document.storageConnection();
        auto mainThreadCallback = [callbackIdentifier, contextIdentifier](auto&& result) mutable {
            ScriptExecutionContext::postTaskTo(contextIdentifier, [callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                downcast<WorkerGlobalScope>(scope).storageConnection().didGetDirectory(callbackIdentifier, WTFMove(result));
            });
        };
        if (!mainThreadConnection)
            return mainThreadCallback(Exception { ExceptionCode::InvalidStateError });

        mainThreadConnection->fileSystemGetDirectory(WTFMove(origin), WTFMove(mainThreadCallback));
    });
}

void WorkerStorageConnection::didGetDirectory(uint64_t callbackIdentifier, ExceptionOr<StorageConnection::DirectoryInfo>&& result)
{
    RefPtr<FileSystemStorageConnection> mainThreadFileSystemStorageConnection = result.hasException() ? nullptr : result.returnValue().second;
    auto releaseConnectionScope = makeScopeExit([connection = mainThreadFileSystemStorageConnection]() mutable {
        if (connection)
            callOnMainThread([connection = WTFMove(connection)]() { });
    });

    auto callback = m_getDirectoryCallbacks.take(callbackIdentifier);
    if (!callback)
        return;

    if (result.hasException())
        return callback(WTFMove(result));

    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });
    releaseConnectionScope.release();

    auto& workerFileSystemStorageConnection = m_scope->getFileSystemStorageConnection(Ref { *mainThreadFileSystemStorageConnection });
    callback(StorageConnection::DirectoryInfo { result.returnValue().first, Ref { workerFileSystemStorageConnection } });
}

} // namespace WebCore
