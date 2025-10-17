/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
#include "FileSystemDirectoryHandle.h"

#include "ContextDestructionObserverInlines.h"
#include "FileSystemHandleCloseScope.h"
#include "FileSystemStorageConnection.h"
#include "JSDOMPromiseDeferred.h"
#include "JSFileSystemDirectoryHandle.h"
#include "JSFileSystemFileHandle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FileSystemDirectoryHandle);

Ref<FileSystemDirectoryHandle> FileSystemDirectoryHandle::create(ScriptExecutionContext& context, String&& name, FileSystemHandleIdentifier identifier, Ref<FileSystemStorageConnection>&& connection)
{
    auto result = adoptRef(*new FileSystemDirectoryHandle(context, WTFMove(name), identifier, WTFMove(connection)));
    result->suspendIfNeeded();
    return result;
}

FileSystemDirectoryHandle::FileSystemDirectoryHandle(ScriptExecutionContext& context, String&& name, FileSystemHandleIdentifier identifier, Ref<FileSystemStorageConnection>&& connection)
    : FileSystemHandle(context, FileSystemHandle::Kind::Directory, WTFMove(name), identifier, WTFMove(connection))
{
}

void FileSystemDirectoryHandle::getFileHandle(const String& name, const FileSystemDirectoryHandle::GetFileOptions& options, DOMPromiseDeferred<IDLInterface<FileSystemFileHandle>>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().getFileHandle(identifier(), name, options.create, [weakContext = WeakPtr { *scriptExecutionContext() }, connection = Ref { connection() }, name, promise = WTFMove(promise)](auto result) mutable {
        if (result.hasException())
            return promise.reject(result.releaseException());

        RefPtr context = weakContext.get();
        if (!context)
            return promise.reject(Exception { ExceptionCode::InvalidStateError, "Context has stopped"_s });

        auto [identifier, isDirectory] = result.returnValue()->release();
        ASSERT(!isDirectory);
        promise.resolve(FileSystemFileHandle::create(*context, String { name }, identifier, WTFMove(connection)));
    });
}

void FileSystemDirectoryHandle::getDirectoryHandle(const String& name, const FileSystemDirectoryHandle::GetDirectoryOptions& options, DOMPromiseDeferred<IDLInterface<FileSystemDirectoryHandle>>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().getDirectoryHandle(identifier(), name, options.create, [weakContext = WeakPtr { *scriptExecutionContext() }, connection = Ref { connection() }, name, promise = WTFMove(promise)](auto result) mutable {
        if (result.hasException())
            return promise.reject(result.releaseException());

        RefPtr context = weakContext.get();
        if (!context)
            return promise.reject(Exception { ExceptionCode::InvalidStateError, "Context has stopped"_s });

        auto [identifier, isDirectory] = result.returnValue()->release();
        ASSERT(isDirectory);
        promise.resolve(FileSystemDirectoryHandle::create(*context, String { name }, identifier, WTFMove(connection)));
    });
}

void FileSystemDirectoryHandle::removeEntry(const String& name, const FileSystemDirectoryHandle::RemoveOptions& options, DOMPromiseDeferred<void>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().removeEntry(identifier(), name, options.recursive, [promise = WTFMove(promise)](auto result) mutable {
        promise.settle(WTFMove(result));
    });
}

void FileSystemDirectoryHandle::resolve(const FileSystemHandle& handle, DOMPromiseDeferred<IDLSequence<IDLUSVString>>&& promise)
{
    if (isClosed())
        return promise.reject(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().resolve(identifier(), handle.identifier(), [promise = WTFMove(promise)](auto result) mutable {
        promise.settle(WTFMove(result));
    });
}

void FileSystemDirectoryHandle::getHandleNames(CompletionHandler<void(ExceptionOr<Vector<String>>&&)>&& completionHandler)
{
    if (isClosed())
        return completionHandler(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().getHandleNames(identifier(), WTFMove(completionHandler));
}

void FileSystemDirectoryHandle::getHandle(const String& name, CompletionHandler<void(ExceptionOr<Ref<FileSystemHandle>>&&)>&& completionHandler)
{
    if (isClosed())
        return completionHandler(Exception { ExceptionCode::InvalidStateError, "Handle is closed"_s });

    connection().getHandle(identifier(), name, [weakContext = WeakPtr { *scriptExecutionContext() }, name, connection = Ref { connection() }, completionHandler = WTFMove(completionHandler)](auto result) mutable {
        if (result.hasException())
            return completionHandler(result.releaseException());

        auto [identifier, isDirectory] = result.returnValue()->release();
        RefPtr context = weakContext.get();
        if (!context)
            return completionHandler(Exception { ExceptionCode::InvalidStateError, "Context has stopped"_s });

        if (isDirectory) {
            Ref<FileSystemHandle> handle = FileSystemDirectoryHandle::create(*context, String { name }, identifier, WTFMove(connection));
            return completionHandler(WTFMove(handle));
        }

        Ref<FileSystemHandle> handle = FileSystemFileHandle::create(*context, String { name }, identifier, WTFMove(connection));
        completionHandler(WTFMove(handle));
    });
}

using FileSystemDirectoryHandleIterator = FileSystemDirectoryHandle::Iterator;

Ref<FileSystemDirectoryHandleIterator> FileSystemDirectoryHandle::createIterator(ScriptExecutionContext*)
{
    return Iterator::create(*this);
}

Ref<FileSystemDirectoryHandleIterator> FileSystemDirectoryHandleIterator::create(FileSystemDirectoryHandle& source)
{
    return adoptRef(*new FileSystemDirectoryHandle::Iterator(source));
}

void FileSystemDirectoryHandleIterator::next(CompletionHandler<void(ExceptionOr<Result>&&)>&& completionHandler)
{
    ASSERT(!m_isWaitingForResult);
    m_isWaitingForResult = true;

    auto wrappedCompletionHandler = [protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)](auto result) mutable {
        protectedThis->m_isWaitingForResult = false;
        completionHandler(WTFMove(result));
    };

    if (!m_isInitialized) {
        m_source->getHandleNames([protectedThis = Ref { *this }, completionHandler = WTFMove(wrappedCompletionHandler)](auto result) mutable {
            protectedThis->m_isInitialized = true;
            if (result.hasException())
                return completionHandler(result.releaseException());

            protectedThis->m_keys = result.releaseReturnValue();
            protectedThis->advance(WTFMove(completionHandler));
        });
        return;
    }

    advance(WTFMove(wrappedCompletionHandler));
}

void FileSystemDirectoryHandleIterator::advance(CompletionHandler<void(ExceptionOr<Result>&&)>&& completionHandler)
{
    ASSERT(m_isInitialized);

    if (m_index >= m_keys.size()) {
        Result result = std::nullopt;
        return completionHandler(Result { });
    }

    auto key = m_keys[m_index++];
    m_source->getHandle(key, [protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler), key](auto result) mutable {
        if (result.hasException()) {
            if (result.exception().code() == ExceptionCode::NotFoundError)
                return protectedThis->advance(WTFMove(completionHandler));

            return completionHandler(result.releaseException());
        }

        Result resultValue = KeyValuePair<String, Ref<FileSystemHandle>> { WTFMove(key), result.releaseReturnValue() };
        completionHandler(WTFMove(resultValue));
    });
}

} // namespace WebCore


