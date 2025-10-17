/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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

#include "Connection.h"
#include "FileSystemStorageError.h"
#include <WebCore/FileSystemStorageConnection.h>

namespace IPC {
class Connection;

template<> struct AsyncReplyError<Expected<WebCore::FileSystemHandleIdentifier, WebKit::FileSystemStorageError>> {
    static Expected<WebCore::FileSystemHandleIdentifier, WebKit::FileSystemStorageError> create()
    {
        return makeUnexpected(WebKit::FileSystemStorageError::Unknown);
    }
};

template<> struct AsyncReplyError<Expected<Vector<String>, WebKit::FileSystemStorageError>> {
    static Expected<Vector<String>, WebKit::FileSystemStorageError> create()
    {
        return makeUnexpected(WebKit::FileSystemStorageError::Unknown);
    }
};

template<> struct AsyncReplyError<Expected<std::pair<String, bool>, WebKit::FileSystemStorageError>> {
    static Expected<std::pair<String, bool>, WebKit::FileSystemStorageError> create()
    {
        return makeUnexpected(WebKit::FileSystemStorageError::Unknown);
    }
};

}

namespace WebCore {
template<typename> class ExceptionOr;
class FileSystemDirectoryHandle;
class FileSystemFileHandle;
}

namespace WebKit {

class WebFileSystemStorageConnection final : public WebCore::FileSystemStorageConnection {
public:
    static Ref<WebFileSystemStorageConnection> create(Ref<IPC::Connection>&&);
    void connectionClosed();
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

private:
    explicit WebFileSystemStorageConnection(Ref<IPC::Connection>&&);

    // FileSystemStorageConnection
    void closeHandle(WebCore::FileSystemHandleIdentifier) final;
    void isSameEntry(WebCore::FileSystemHandleIdentifier, WebCore::FileSystemHandleIdentifier, WebCore::FileSystemStorageConnection::SameEntryCallback&&) final;
    void move(WebCore::FileSystemHandleIdentifier, WebCore::FileSystemHandleIdentifier, const String& newName, VoidCallback&&) final;
    void getFileHandle(WebCore::FileSystemHandleIdentifier, const String& name, bool createIfNecessary, WebCore::FileSystemStorageConnection::GetHandleCallback&&) final;
    void getDirectoryHandle(WebCore::FileSystemHandleIdentifier, const String& name, bool createIfNecessary, WebCore::FileSystemStorageConnection::GetHandleCallback&&) final;
    void removeEntry(WebCore::FileSystemHandleIdentifier, const String& name, bool deleteRecursively, WebCore::FileSystemStorageConnection::VoidCallback&&) final;
    void resolve(WebCore::FileSystemHandleIdentifier, WebCore::FileSystemHandleIdentifier, WebCore::FileSystemStorageConnection::ResolveCallback&&) final;
    void getHandleNames(WebCore::FileSystemHandleIdentifier, FileSystemStorageConnection::GetHandleNamesCallback&&) final;
    void getHandle(WebCore::FileSystemHandleIdentifier, const String& name, FileSystemStorageConnection::GetHandleCallback&&) final;
    void getFile(WebCore::FileSystemHandleIdentifier, StringCallback&&) final;

    void createSyncAccessHandle(WebCore::FileSystemHandleIdentifier, WebCore::FileSystemStorageConnection::GetAccessHandleCallback&&) final;
    void closeSyncAccessHandle(WebCore::FileSystemHandleIdentifier, WebCore::FileSystemSyncAccessHandleIdentifier, EmptyCallback&&) final;
    void requestNewCapacityForSyncAccessHandle(WebCore::FileSystemHandleIdentifier, WebCore::FileSystemSyncAccessHandleIdentifier, uint64_t newCapacity, RequestCapacityCallback&& completionHandler) final;
    void registerSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier, WebCore::ScriptExecutionContextIdentifier) final;
    void unregisterSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier) final;
    void invalidateAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier) final;
    void createWritable(WebCore::FileSystemHandleIdentifier, bool keepExistingData, WebCore::FileSystemStorageConnection::VoidCallback&&) final;
    void closeWritable(WebCore::FileSystemHandleIdentifier, WebCore::FileSystemWriteCloseReason, WebCore::FileSystemStorageConnection::VoidCallback&&) final;
    void executeCommandForWritable(WebCore::FileSystemHandleIdentifier, WebCore::FileSystemWriteCommandType, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError, WebCore::FileSystemStorageConnection::VoidCallback&&) final;

    HashMap<WebCore::FileSystemSyncAccessHandleIdentifier, WebCore::ScriptExecutionContextIdentifier> m_syncAccessHandles;
    RefPtr<IPC::Connection> m_connection;
};

} // namespace WebKit
