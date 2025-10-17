/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
#include "WorkerFileSystemStorageConnection.h"

#include "FileSystemHandleCloseScope.h"
#include "FileSystemSyncAccessHandle.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerThread.h"
#include <wtf/Scope.h>

namespace WebCore {

Ref<WorkerFileSystemStorageConnection> WorkerFileSystemStorageConnection::create(WorkerGlobalScope& scope, Ref<FileSystemStorageConnection>&& mainThreadConnection)
{
    return adoptRef(*new WorkerFileSystemStorageConnection(scope, WTFMove(mainThreadConnection)));
}

WorkerFileSystemStorageConnection::WorkerFileSystemStorageConnection(WorkerGlobalScope& scope, Ref<FileSystemStorageConnection>&& connection)
    : m_scope(scope)
    , m_mainThreadConnection(WTFMove(connection))
{
}

WorkerFileSystemStorageConnection::~WorkerFileSystemStorageConnection()
{
    if (m_mainThreadConnection)
        callOnMainThread([mainThreadConenction = WTFMove(m_mainThreadConnection)]() { });
}

void WorkerFileSystemStorageConnection::connectionClosed()
{
    for (auto handle : m_syncAccessHandles.values())
        handle->invalidate();

    scopeClosed();
}

void WorkerFileSystemStorageConnection::scopeClosed()
{
    auto sameEntryCallbacks = std::exchange(m_sameEntryCallbacks, { });
    for (auto& callback : sameEntryCallbacks.values())
        callback(Exception { ExceptionCode::InvalidStateError });

    auto getHandleCallbacks = std::exchange(m_getHandleCallbacks, { });
    for (auto& callback : getHandleCallbacks.values())
        callback(Exception { ExceptionCode::InvalidStateError });

    auto removeEntryCallbacks = std::exchange(m_voidCallbacks, { });
    for (auto& callback : removeEntryCallbacks.values())
        callback(Exception { ExceptionCode::InvalidStateError });

    auto resolveCallbacks = std::exchange(m_resolveCallbacks, { });
    for (auto& callback : resolveCallbacks.values())
        callback(Exception { ExceptionCode::InvalidStateError });

    auto stringCallbacks = std::exchange(m_stringCallbacks, { });
    for (auto& callback : stringCallbacks.values())
        callback(Exception { ExceptionCode::InvalidStateError });

    m_scope = nullptr;
}

void WorkerFileSystemStorageConnection::closeHandle(FileSystemHandleIdentifier identifier)
{
    callOnMainThread([mainThreadConnection = m_mainThreadConnection, identifier]() mutable {
        if (mainThreadConnection)
            mainThreadConnection->closeHandle(identifier);
    });
}

void WorkerFileSystemStorageConnection::isSameEntry(FileSystemHandleIdentifier identifier, FileSystemHandleIdentifier otherIdentifier, FileSystemStorageConnection::SameEntryCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_sameEntryCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, otherIdentifier]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](ExceptionOr<bool>&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->didIsSameEntry(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->isSameEntry(identifier, otherIdentifier, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::didIsSameEntry(CallbackIdentifier callbackIdentifier, ExceptionOr<bool>&& result)
{
    if (auto callback = m_sameEntryCallbacks.take(callbackIdentifier))
        callback(WTFMove(result));
}

void WorkerFileSystemStorageConnection::getFileHandle(FileSystemHandleIdentifier identifier, const String& name, bool createIfNecessary, FileSystemStorageConnection::GetHandleCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_getHandleCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, name = name.isolatedCopy(), createIfNecessary]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->didGetHandle(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->getFileHandle(identifier, name, createIfNecessary, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::getDirectoryHandle(FileSystemHandleIdentifier identifier, const String& name, bool createIfNecessary, FileSystemStorageConnection::GetHandleCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_getHandleCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, name = name.isolatedCopy(), createIfNecessary]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->didGetHandle(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->getDirectoryHandle(identifier, name, createIfNecessary, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::didGetHandle(CallbackIdentifier callbackIdentifier, ExceptionOr<Ref<FileSystemHandleCloseScope>>&& result)
{
    if (auto callback = m_getHandleCallbacks.take(callbackIdentifier))
        callback(WTFMove(result));
}

void WorkerFileSystemStorageConnection::removeEntry(FileSystemHandleIdentifier identifier, const String& name, bool deleteRecursively, FileSystemStorageConnection::VoidCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_voidCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, name = name.isolatedCopy(), deleteRecursively]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->completeVoidCallback(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->removeEntry(identifier, name, deleteRecursively, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::resolve(FileSystemHandleIdentifier identifier, FileSystemHandleIdentifier otherIdentifier, FileSystemStorageConnection::ResolveCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_resolveCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, otherIdentifier]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->didResolve(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->resolve(identifier, otherIdentifier, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::didResolve(CallbackIdentifier callbackIdentifier, ExceptionOr<Vector<String>>&& result)
{
    if (auto callback = m_resolveCallbacks.take(callbackIdentifier))
        callback(WTFMove(result));
}

void WorkerFileSystemStorageConnection::getFile(FileSystemHandleIdentifier identifier, StringCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_stringCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->completeStringCallback(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->getFile(identifier, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::completeStringCallback(CallbackIdentifier callbackIdentifier, ExceptionOr<String>&& result)
{
    if (auto callback = m_stringCallbacks.take(callbackIdentifier))
        callback(WTFMove(result));
}

void WorkerFileSystemStorageConnection::didCreateSyncAccessHandle(CallbackIdentifier callbackIdentifier, ExceptionOr<FileSystemStorageConnection::SyncAccessHandleInfo>&& result)
{
    if (auto callback = m_getAccessHandlCallbacks.take(callbackIdentifier))
        callback(WTFMove(result));
}

void WorkerFileSystemStorageConnection::completeVoidCallback(CallbackIdentifier callbackIdentifier, ExceptionOr<void>&& result)
{
    if (auto callback = m_voidCallbacks.take(callbackIdentifier))
        callback(WTFMove(result));
}

void WorkerFileSystemStorageConnection::createSyncAccessHandle(FileSystemHandleIdentifier identifier, FileSystemStorageConnection::GetAccessHandleCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_getAccessHandlCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->didCreateSyncAccessHandle(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->createSyncAccessHandle(identifier, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::closeSyncAccessHandle(FileSystemHandleIdentifier identifier, FileSystemSyncAccessHandleIdentifier accessHandleIdentifier)
{
    if (!m_scope)
        return;

    BinarySemaphore semaphore;
    callOnMainThread([mainThreadConnection = m_mainThreadConnection, identifier, accessHandleIdentifier, &semaphore]() mutable {
        mainThreadConnection->closeSyncAccessHandle(identifier, accessHandleIdentifier, [&semaphore]() {
            semaphore.signal();
        });
    });
    semaphore.wait();
}

void WorkerFileSystemStorageConnection::closeSyncAccessHandle(FileSystemHandleIdentifier, FileSystemSyncAccessHandleIdentifier, EmptyCallback&&)
{
    ASSERT_NOT_REACHED();
}

void WorkerFileSystemStorageConnection::registerSyncAccessHandle(FileSystemSyncAccessHandleIdentifier identifier, FileSystemSyncAccessHandle& handle)
{
    if (!m_scope)
        return;

    m_syncAccessHandles.add(identifier, WeakPtr { handle });
    callOnMainThread([identifier, contextIdentifier = m_scope->identifier(), mainThreadConnection = m_mainThreadConnection]() mutable {
        mainThreadConnection->registerSyncAccessHandle(identifier, contextIdentifier);
    });
}

void WorkerFileSystemStorageConnection::unregisterSyncAccessHandle(FileSystemSyncAccessHandleIdentifier identifier)
{
    m_syncAccessHandles.remove(identifier);
    callOnMainThread([identifier, mainThreadConnection = m_mainThreadConnection]() mutable {
        mainThreadConnection->unregisterSyncAccessHandle(identifier);
    });
}

void WorkerFileSystemStorageConnection::invalidateAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier identifier)
{
    if (auto handle = m_syncAccessHandles.get(identifier))
        handle->invalidate();
}

void WorkerFileSystemStorageConnection::createWritable(FileSystemHandleIdentifier identifier, bool keepExistingData, VoidCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_voidCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, keepExistingData]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (RefPtr connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->completeVoidCallback(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->createWritable(identifier, keepExistingData, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::closeWritable(FileSystemHandleIdentifier identifier, FileSystemWriteCloseReason reason, VoidCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_voidCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, reason]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (RefPtr connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->completeVoidCallback(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->closeWritable(identifier, reason, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::executeCommandForWritable(FileSystemHandleIdentifier identifier, FileSystemWriteCommandType type, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError, VoidCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_voidCallbacks.add(callbackIdentifier, WTFMove(callback));

    Vector<uint8_t> bytes;
    bytes.append(dataBytes);
    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, type, position, size, bytes = WTFMove(bytes), hasDataError]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (RefPtr connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->completeVoidCallback(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->executeCommandForWritable(identifier, type, position, size, bytes.span(), hasDataError, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::getHandleNames(FileSystemHandleIdentifier identifier, GetHandleNamesCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_getHandleNamesCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->didGetHandleNames(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->getHandleNames(identifier, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::didGetHandleNames(CallbackIdentifier callbackIdentifier, ExceptionOr<Vector<String>>&& result)
{
    if (auto callback = m_getHandleNamesCallbacks.take(callbackIdentifier))
        callback(WTFMove(result));
}

void WorkerFileSystemStorageConnection::getHandle(FileSystemHandleIdentifier identifier, const String& name, GetHandleCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_getHandleCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, name = name.isolatedCopy()]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->didGetHandle(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->getHandle(identifier, name, WTFMove(mainThreadCallback));
    });
}

void WorkerFileSystemStorageConnection::move(FileSystemHandleIdentifier identifier, FileSystemHandleIdentifier destinationIdentifier, const String& newName, VoidCallback&& callback)
{
    if (!m_scope)
        return callback(Exception { ExceptionCode::InvalidStateError });

    auto callbackIdentifier = CallbackIdentifier::generate();
    m_voidCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([callbackIdentifier, workerThread = Ref { m_scope->thread() }, mainThreadConnection = m_mainThreadConnection, identifier, destinationIdentifier, name = crossThreadCopy(newName)]() mutable {
        auto mainThreadCallback = [callbackIdentifier, workerThread = WTFMove(workerThread)](auto&& result) mutable {
            workerThread->runLoop().postTaskForMode([callbackIdentifier, result = crossThreadCopy(WTFMove(result))] (auto& scope) mutable {
                if (auto connection = downcast<WorkerGlobalScope>(scope).fileSystemStorageConnection())
                    connection->completeVoidCallback(callbackIdentifier, WTFMove(result));
            }, WorkerRunLoop::defaultMode());
        };

        mainThreadConnection->move(identifier, destinationIdentifier, name, WTFMove(mainThreadCallback));
    });
}

std::optional<uint64_t> WorkerFileSystemStorageConnection::requestNewCapacityForSyncAccessHandle(FileSystemHandleIdentifier identifier, FileSystemSyncAccessHandleIdentifier accessHandleIdentifier, uint64_t newCapacity)
{
    if (!m_scope)
        return std::nullopt;

    if (!m_mainThreadConnection)
        return std::nullopt;

    BinarySemaphore semaphore;
    std::optional<uint64_t> grantedCapacity;
    callOnMainThread([mainThreadConnection = m_mainThreadConnection, identifier, accessHandleIdentifier, newCapacity, &grantedCapacity, &semaphore]() {
        auto mainThreadCallback = [&grantedCapacity, &semaphore](auto result) {
            grantedCapacity = result;
            semaphore.signal();
        };
        mainThreadConnection->requestNewCapacityForSyncAccessHandle(identifier, accessHandleIdentifier, newCapacity, WTFMove(mainThreadCallback));
    });
    semaphore.wait();
    return grantedCapacity;
}

void WorkerFileSystemStorageConnection::requestNewCapacityForSyncAccessHandle(FileSystemHandleIdentifier, FileSystemSyncAccessHandleIdentifier, uint64_t, RequestCapacityCallback&& callback)
{
    ASSERT_NOT_REACHED();
    return callback(std::nullopt);
}

} // namespace WebCore
