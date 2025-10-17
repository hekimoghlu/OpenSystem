/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#include "FileSystemHandle.h"

#include "FileSystemStorageConnection.h"
#include "JSDOMPromiseDeferred.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FileSystemHandle);

FileSystemHandle::FileSystemHandle(ScriptExecutionContext& context, FileSystemHandle::Kind kind, String&& name, FileSystemHandleIdentifier identifier, Ref<FileSystemStorageConnection>&& connection)
    : ActiveDOMObject(&context)
    , m_kind(kind)
    , m_name(WTFMove(name))
    , m_identifier(identifier)
    , m_connection(WTFMove(connection))
{
}

FileSystemHandle::~FileSystemHandle()
{
    close();
}

void FileSystemHandle::close()
{
    if (m_isClosed)
        return;
    
    m_isClosed = true;
    m_connection->closeHandle(m_identifier);
}

void FileSystemHandle::isSameEntry(FileSystemHandle& handle, DOMPromiseDeferred<IDLBoolean>&& promise) const
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    if (m_kind != handle.kind() || m_name != handle.name())
        return promise.resolve(false);

    m_connection->isSameEntry(m_identifier, handle.identifier(), [promise = WTFMove(promise)](auto result) mutable {
        promise.settle(WTFMove(result));
    });
}

void FileSystemHandle::move(FileSystemHandle& destinationHandle, const String& newName, DOMPromiseDeferred<void>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    if (destinationHandle.kind() != Kind::Directory)
        return promise.reject(Exception { ExceptionCode::TypeMismatchError });

    m_connection->move(m_identifier, destinationHandle.identifier(), newName, [this, protectedThis = Ref { *this }, newName, promise = WTFMove(promise)](auto result) mutable {
        if (!result.hasException())
            m_name = newName;

        promise.settle(WTFMove(result));
    });
}

void FileSystemHandle::stop()
{
    close();
}

} // namespace WebCore
