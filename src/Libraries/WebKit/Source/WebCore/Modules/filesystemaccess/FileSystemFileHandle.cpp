/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#include "FileSystemFileHandle.h"

#include "ContextDestructionObserverInlines.h"
#include "File.h"
#include "FileSystemHandleCloseScope.h"
#include "FileSystemStorageConnection.h"
#include "FileSystemSyncAccessHandle.h"
#include "FileSystemWritableFileStream.h"
#include "FileSystemWritableFileStreamSink.h"
#include "JSDOMPromiseDeferred.h"
#include "JSFile.h"
#include "JSFileSystemSyncAccessHandle.h"
#include "JSFileSystemWritableFileStream.h"
#include "WorkerFileSystemStorageConnection.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FileSystemFileHandle);

Ref<FileSystemFileHandle> FileSystemFileHandle::create(ScriptExecutionContext& context, String&& name, FileSystemHandleIdentifier identifier, Ref<FileSystemStorageConnection>&& connection)
{
    auto result = adoptRef(*new FileSystemFileHandle(context, WTFMove(name), identifier, WTFMove(connection)));
    result->suspendIfNeeded();
    return result;
}

FileSystemFileHandle::FileSystemFileHandle(ScriptExecutionContext& context, String&& name, FileSystemHandleIdentifier identifier, Ref<FileSystemStorageConnection>&& connection)
    : FileSystemHandle(context, FileSystemHandle::Kind::File, WTFMove(name), identifier, WTFMove(connection))
{
}

void FileSystemFileHandle::getFile(DOMPromiseDeferred<IDLInterface<File>>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().getFile(identifier(), [protectedThis = Ref { *this }, promise = WTFMove(promise)](auto result) mutable {
        if (result.hasException())
            return promise.reject(result.releaseException());

        RefPtr context = protectedThis->scriptExecutionContext();
        if (!context)
            return promise.reject(Exception { ExceptionCode::InvalidStateError, "Context has stopped"_s });

        promise.resolve(File::create(context.get(), result.returnValue(), { }, protectedThis->name()));
    });
}

void FileSystemFileHandle::createSyncAccessHandle(DOMPromiseDeferred<IDLInterface<FileSystemSyncAccessHandle>>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().createSyncAccessHandle(identifier(), [protectedThis = Ref { *this }, promise = WTFMove(promise)](auto result) mutable {
        if (result.hasException())
            return promise.reject(result.releaseException());

        auto info = result.releaseReturnValue();
        if (!info.file)
            return promise.reject(Exception { ExceptionCode::UnknownError, "Invalid platform file handle"_s });

        RefPtr context = protectedThis->scriptExecutionContext();
        if (!context) {
            protectedThis->closeSyncAccessHandle(info.identifier);
            return promise.reject(Exception { ExceptionCode::InvalidStateError, "Context has stopped"_s });
        }

        promise.resolve(FileSystemSyncAccessHandle::create(*context, protectedThis.get(), info.identifier, WTFMove(info.file), info.capacity));
    });
}

void FileSystemFileHandle::closeSyncAccessHandle(FileSystemSyncAccessHandleIdentifier accessHandleIdentifier)
{
    if (isClosed())
        return;

    downcast<WorkerFileSystemStorageConnection>(connection()).closeSyncAccessHandle(identifier(), accessHandleIdentifier);
}

std::optional<uint64_t> FileSystemFileHandle::requestNewCapacityForSyncAccessHandle(FileSystemSyncAccessHandleIdentifier accessHandleIdentifier, uint64_t newCapacity)
{
    if (isClosed())
        return std::nullopt;

    return downcast<WorkerFileSystemStorageConnection>(connection()).requestNewCapacityForSyncAccessHandle(identifier(), accessHandleIdentifier, newCapacity);
}

void FileSystemFileHandle::registerSyncAccessHandle(FileSystemSyncAccessHandleIdentifier identifier, FileSystemSyncAccessHandle& handle)
{
    if (isClosed())
        return;

    downcast<WorkerFileSystemStorageConnection>(connection()).registerSyncAccessHandle(identifier, handle);
}

void FileSystemFileHandle::unregisterSyncAccessHandle(FileSystemSyncAccessHandleIdentifier identifier)
{
    if (isClosed())
        return;

    connection().unregisterSyncAccessHandle(identifier);
}

// https://fs.spec.whatwg.org/#api-filesystemfilehandle-createwritable
void FileSystemFileHandle::createWritable(const CreateWritableOptions& options, DOMPromiseDeferred<IDLInterface<FileSystemWritableFileStream>>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().createWritable(identifier(), options.keepExistingData, [this, protectedThis = Ref { *this }, promise = WTFMove(promise)](auto result) mutable {
        if (result.hasException())
            return promise.reject(result.releaseException());

        RefPtr context = protectedThis->scriptExecutionContext();
        if (!context) {
            closeWritable(FileSystemWriteCloseReason::Aborted);
            return promise.reject(Exception { ExceptionCode::InvalidStateError, "Context has stopped"_s });
        }

        auto* globalObject = JSC::jsCast<JSDOMGlobalObject*>(context->globalObject());
        if (!globalObject) {
            closeWritable(FileSystemWriteCloseReason::Aborted);
            return promise.reject(Exception { ExceptionCode::InvalidStateError, "Global object is invalid"_s });
        }

        auto sink = FileSystemWritableFileStreamSink::create(*this);
        if (sink.hasException()) {
            closeWritable(FileSystemWriteCloseReason::Aborted);
            return promise.reject(sink.releaseException());
        }

        ExceptionOr<Ref<FileSystemWritableFileStream>> stream { Exception { ExceptionCode::UnknownError } };
        {
            // FIXME: Make WritableStream function acquire lock as needed and remove this.
            Locker<JSC::JSLock> locker(globalObject->vm().apiLock());
            stream = FileSystemWritableFileStream::create(*globalObject, sink.releaseReturnValue());
        }

        promise.settle(WTFMove(stream));
    });
}

void FileSystemFileHandle::closeWritable(FileSystemWriteCloseReason reason)
{
    if (!isClosed())
        connection().closeWritable(identifier(), reason, [](auto) { });
}

void FileSystemFileHandle::executeCommandForWritable(FileSystemWriteCommandType type, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError, DOMPromiseDeferred<void>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().executeCommandForWritable(identifier(), type, position, size, dataBytes, hasDataError, [promise = WTFMove(promise)](auto result) mutable {
        // Writable should be closed when stream is closed or errored, and stream will be errored after a failed write.
        promise.settle(WTFMove(result));
    });
}

} // namespace WebCore

