/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
#include "WebFileSystemStorageConnection.h"

#include "NetworkStorageManagerMessages.h"
#include <WebCore/ExceptionOr.h>
#include <WebCore/FileSystemDirectoryHandle.h>
#include <WebCore/FileSystemFileHandle.h>
#include <WebCore/FileSystemHandleCloseScope.h>
#include <WebCore/ScriptExecutionContext.h>
#include <WebCore/WorkerFileSystemStorageConnection.h>
#include <WebCore/WorkerGlobalScope.h>

namespace WebKit {

Ref<WebFileSystemStorageConnection> WebFileSystemStorageConnection::create(Ref<IPC::Connection>&& connection)
{
    return adoptRef(*new WebFileSystemStorageConnection(WTFMove(connection)));
}

WebFileSystemStorageConnection::WebFileSystemStorageConnection(Ref<IPC::Connection>&& connection)
    : m_connection(WTFMove(connection))
{
}

void WebFileSystemStorageConnection::connectionClosed()
{
    m_connection = nullptr;

    for (auto identifier : m_syncAccessHandles.keys())
        invalidateAccessHandle(identifier);
}

void WebFileSystemStorageConnection::closeHandle(WebCore::FileSystemHandleIdentifier identifier)
{
    if (!m_connection)
        return;

    m_connection->send(Messages::NetworkStorageManager::CloseHandle(identifier), 0);
}

void WebFileSystemStorageConnection::isSameEntry(WebCore::FileSystemHandleIdentifier identifier, WebCore::FileSystemHandleIdentifier otherIdentifier, WebCore::FileSystemStorageConnection::SameEntryCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    if (identifier == otherIdentifier)
        return completionHandler(true);

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::IsSameEntry(identifier, otherIdentifier), WTFMove(completionHandler));
}

void WebFileSystemStorageConnection::getFileHandle(WebCore::FileSystemHandleIdentifier identifier, const String& name, bool createIfNecessary, WebCore::FileSystemStorageConnection::GetHandleCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::GetFileHandle(identifier, name, createIfNecessary), [this, protectedThis = Ref { *this }, name, completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(convertToException(result.error()));

        completionHandler(WebCore::FileSystemHandleCloseScope::create(result.value(), false, *this));
    });
}

void WebFileSystemStorageConnection::getDirectoryHandle(WebCore::FileSystemHandleIdentifier identifier, const String& name, bool createIfNecessary, WebCore::FileSystemStorageConnection::GetHandleCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::GetDirectoryHandle(identifier, name, createIfNecessary), [this, protectedThis = Ref { *this }, name, completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(convertToException(result.error()));

        completionHandler(WebCore::FileSystemHandleCloseScope::create(result.value(), true, *this));
    });
}

void WebFileSystemStorageConnection::removeEntry(WebCore::FileSystemHandleIdentifier identifier, const String& name, bool deleteRecursively, WebCore::FileSystemStorageConnection::VoidCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::RemoveEntry(identifier, name, deleteRecursively), [completionHandler = WTFMove(completionHandler)](auto error) mutable {
        return completionHandler(convertToExceptionOr(error));
    });
}

void WebFileSystemStorageConnection::resolve(WebCore::FileSystemHandleIdentifier identifier, WebCore::FileSystemHandleIdentifier otherIdentifier, WebCore::FileSystemStorageConnection::ResolveCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::Resolve(identifier, otherIdentifier), [completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(convertToException(result.error()));

        completionHandler(WTFMove(result.value()));
    });
}

void WebFileSystemStorageConnection::getFile(WebCore::FileSystemHandleIdentifier identifier, StringCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::GetFile(identifier), [completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(convertToException(result.error()));

        completionHandler(WTFMove(result.value()));
    });
}

void WebFileSystemStorageConnection::createSyncAccessHandle(WebCore::FileSystemHandleIdentifier identifier, WebCore::FileSystemStorageConnection::GetAccessHandleCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::CreateSyncAccessHandle(identifier), [completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(convertToException(result.error()));

        completionHandler(WebCore::FileSystemStorageConnection::SyncAccessHandleInfo { *result->identifier, result->handle.release(), result->capacity });
    });
}

void WebFileSystemStorageConnection::closeSyncAccessHandle(WebCore::FileSystemHandleIdentifier identifier, WebCore::FileSystemSyncAccessHandleIdentifier accessHandleIdentifier, FileSystemStorageConnection::EmptyCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler();

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::CloseSyncAccessHandle(identifier, accessHandleIdentifier), WTFMove(completionHandler));
}

void WebFileSystemStorageConnection::getHandleNames(WebCore::FileSystemHandleIdentifier identifier, FileSystemStorageConnection::GetHandleNamesCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::GetHandleNames(identifier), [completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(convertToException(result.error()));

        completionHandler(WTFMove(result.value()));
    });
}

void WebFileSystemStorageConnection::getHandle(WebCore::FileSystemHandleIdentifier identifier, const String& name, FileSystemStorageConnection::GetHandleCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::GetHandle(identifier, name), [this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (!result)
            return completionHandler(convertToException(result.error()));
        
        auto [identifier, isDirectory] = *result.value();
        completionHandler(WebCore::FileSystemHandleCloseScope::create(identifier, isDirectory, *this));
    });
}

void WebFileSystemStorageConnection::move(WebCore::FileSystemHandleIdentifier identifier, WebCore::FileSystemHandleIdentifier destinationIdentifier, const String& newName, VoidCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::Move(identifier, destinationIdentifier, newName), [completionHandler = WTFMove(completionHandler)](auto error) mutable {
        completionHandler(convertToExceptionOr(error));
    });
}

void WebFileSystemStorageConnection::registerSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier identifier, WebCore::ScriptExecutionContextIdentifier contextIdentifier)
{
    m_syncAccessHandles.add(identifier, contextIdentifier);
}

void WebFileSystemStorageConnection::unregisterSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier identifier)
{
    m_syncAccessHandles.remove(identifier);
}

void WebFileSystemStorageConnection::invalidateAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier identifier)
{
    if (auto contextIdentifier = m_syncAccessHandles.get(identifier)) {
        WebCore::ScriptExecutionContext::postTaskTo(contextIdentifier, [identifier](auto& context) mutable {
            if (FileSystemStorageConnection* connection = downcast<WebCore::WorkerGlobalScope>(context).fileSystemStorageConnection())
                connection->invalidateAccessHandle(identifier);
        });
    }
}

void WebFileSystemStorageConnection::createWritable(WebCore::FileSystemHandleIdentifier identifier, bool keepExistingData, WebCore::FileSystemStorageConnection::VoidCallback&& completionHandler)
{
    if (RefPtr connection = m_connection) {
        connection->sendWithAsyncReply(Messages::NetworkStorageManager::CreateWritable(identifier, keepExistingData), [completionHandler = WTFMove(completionHandler)](auto error) mutable {
            completionHandler(convertToExceptionOr(error));
        });
        return;
    }

    completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });
}

void WebFileSystemStorageConnection::closeWritable(WebCore::FileSystemHandleIdentifier identifier, WebCore::FileSystemWriteCloseReason reason, WebCore::FileSystemStorageConnection::VoidCallback&& completionHandler)
{
    if (RefPtr connection = m_connection) {
        connection->sendWithAsyncReply(Messages::NetworkStorageManager::CloseWritable(identifier, reason), [completionHandler = WTFMove(completionHandler)](auto error) mutable {
            completionHandler(convertToExceptionOr(error));
        });
        return;
    }

    completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });
}

void WebFileSystemStorageConnection::executeCommandForWritable(WebCore::FileSystemHandleIdentifier identifier, WebCore::FileSystemWriteCommandType type, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError, WebCore::FileSystemStorageConnection::VoidCallback&& completionHandler)
{
    if (RefPtr connection = m_connection) {
        connection->sendWithAsyncReply(Messages::NetworkStorageManager::ExecuteCommandForWritable(identifier, type, position, size, dataBytes, hasDataError), [completionHandler = WTFMove(completionHandler)](auto error) mutable {
            completionHandler(convertToExceptionOr(error));
        });
        return;
    }

    completionHandler(WebCore::Exception { WebCore::ExceptionCode::UnknownError, "Connection is lost"_s });
}

void WebFileSystemStorageConnection::requestNewCapacityForSyncAccessHandle(WebCore::FileSystemHandleIdentifier identifier, WebCore::FileSystemSyncAccessHandleIdentifier accessHandleIdentifier, uint64_t newCapacity, RequestCapacityCallback&& completionHandler)
{
    if (!m_connection)
        return completionHandler(std::nullopt);

    m_connection->sendWithAsyncReply(Messages::NetworkStorageManager::RequestNewCapacityForSyncAccessHandle(identifier, accessHandleIdentifier, newCapacity), WTFMove(completionHandler));
}

} // namespace WebKit
