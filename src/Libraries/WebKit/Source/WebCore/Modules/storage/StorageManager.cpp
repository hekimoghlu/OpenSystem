/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#include "StorageManager.h"

#include "ClientOrigin.h"
#include "Document.h"
#include "ExceptionOr.h"
#include "FileSystemDirectoryHandle.h"
#include "FileSystemStorageConnection.h"
#include "JSDOMPromiseDeferred.h"
#include "JSFileSystemDirectoryHandle.h"
#include "JSStorageManager.h"
#include "NavigatorBase.h"
#include "SecurityOrigin.h"
#include "WorkerGlobalScope.h"
#include "WorkerStorageConnection.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(StorageManager);

Ref<StorageManager> StorageManager::create(NavigatorBase& navigator)
{
    return adoptRef(*new StorageManager(navigator));
}

StorageManager::StorageManager(NavigatorBase& navigator)
    : m_navigator(navigator)
{
}

StorageManager::~StorageManager() = default;

struct ConnectionInfo {
    StorageConnection& connection;
    ClientOrigin origin;
};

static ExceptionOr<ConnectionInfo> connectionInfo(NavigatorBase* navigator)
{
    if (!navigator)
        return Exception { ExceptionCode::InvalidStateError, "Navigator does not exist"_s };

    RefPtr context = navigator->scriptExecutionContext();
    if (!context)
        return Exception { ExceptionCode::InvalidStateError, "Context is invalid"_s };

    if (context->canAccessResource(ScriptExecutionContext::ResourceType::StorageManager) == ScriptExecutionContext::HasResourceAccess::No)
        return Exception { ExceptionCode::TypeError, "Context not access storage"_s };

    RefPtr origin = context->securityOrigin();
    ASSERT(origin);

    if (RefPtr document = dynamicDowncast<Document>(*context)) {
        if (RefPtr connection = document->storageConnection())
            return ConnectionInfo { *connection, { document->topOrigin().data(), origin->data() } };

        return Exception { ExceptionCode::InvalidStateError, "Connection is invalid"_s };
    }

    if (RefPtr globalScope = dynamicDowncast<WorkerGlobalScope>(*context))
        return ConnectionInfo { globalScope->storageConnection(), { globalScope->topOrigin().data(), origin->data() } };

    return Exception { ExceptionCode::NotSupportedError };
}

void StorageManager::persisted(DOMPromiseDeferred<IDLBoolean>&& promise)
{
    auto connectionInfoOrException = connectionInfo(m_navigator.get());
    if (connectionInfoOrException.hasException())
        return promise.reject(connectionInfoOrException.releaseException());

    auto connectionInfo = connectionInfoOrException.releaseReturnValue();
    connectionInfo.connection.getPersisted(WTFMove(connectionInfo.origin), [promise = WTFMove(promise)](bool persisted) mutable {
        promise.resolve(persisted);
    });
}

void StorageManager::persist(DOMPromiseDeferred<IDLBoolean>&& promise)
{
    auto connectionInfoOrException = connectionInfo(m_navigator.get());
    if (connectionInfoOrException.hasException())
        return promise.reject(connectionInfoOrException.releaseException());

    auto connectionInfo = connectionInfoOrException.releaseReturnValue();
    connectionInfo.connection.persist(connectionInfo.origin, [promise = WTFMove(promise)](bool persisted) mutable {
        promise.resolve(persisted);
    });
}

void StorageManager::estimate(DOMPromiseDeferred<IDLDictionary<StorageEstimate>>&& promise)
{
    auto connectionInfoOrException = connectionInfo(m_navigator.get());
    if (connectionInfoOrException.hasException())
        return promise.reject(connectionInfoOrException.releaseException());

    auto connectionInfo = connectionInfoOrException.releaseReturnValue();
    connectionInfo.connection.getEstimate(WTFMove(connectionInfo.origin), [promise = WTFMove(promise)](ExceptionOr<StorageEstimate>&& result) mutable {
        promise.settle(WTFMove(result));
    });
}

void StorageManager::fileSystemAccessGetDirectory(DOMPromiseDeferred<IDLInterface<FileSystemDirectoryHandle>>&& promise)
{
    auto connectionInfoOrException = connectionInfo(m_navigator.get());
    if (connectionInfoOrException.hasException())
        return promise.reject(connectionInfoOrException.releaseException());

    auto connectionInfo = connectionInfoOrException.releaseReturnValue();
    connectionInfo.connection.fileSystemGetDirectory(WTFMove(connectionInfo.origin), [promise = WTFMove(promise), weakNavigator = m_navigator](auto&& result) mutable {
        if (result.hasException())
            return promise.reject(result.releaseException());

        auto [identifier, connection] = result.releaseReturnValue();
        RefPtr context = weakNavigator ? weakNavigator->scriptExecutionContext() : nullptr;
        if (!context) {
            connection->closeHandle(identifier);
            return promise.reject(Exception { ExceptionCode::InvalidStateError, "Context has stopped"_s });
        }

        promise.resolve(FileSystemDirectoryHandle::create(*context, { }, identifier, Ref { *connection }));
    });
}

} // namespace WebCore
