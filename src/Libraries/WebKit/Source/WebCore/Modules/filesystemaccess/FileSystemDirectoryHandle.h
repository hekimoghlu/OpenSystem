/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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

#include "FileSystemHandle.h"

namespace WebCore {

class FileSystemFileHandle;
template<typename> class ExceptionOr;

class FileSystemDirectoryHandle final : public FileSystemHandle {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FileSystemDirectoryHandle);
public:
    struct GetFileOptions {
        bool create { false };
    };

    struct GetDirectoryOptions {
        bool create { false };
    };
    
    struct RemoveOptions {
        bool recursive { false };
    };

    WEBCORE_EXPORT static Ref<FileSystemDirectoryHandle> create(ScriptExecutionContext&, String&&, FileSystemHandleIdentifier, Ref<FileSystemStorageConnection>&&);
    void getFileHandle(const String& name, const GetFileOptions&, DOMPromiseDeferred<IDLInterface<FileSystemFileHandle>>&&);
    void getDirectoryHandle(const String& name, const GetDirectoryOptions&, DOMPromiseDeferred<IDLInterface<FileSystemDirectoryHandle>>&&);
    void removeEntry(const String& name, const RemoveOptions&, DOMPromiseDeferred<void>&&);
    void resolve(const FileSystemHandle&, DOMPromiseDeferred<IDLSequence<IDLUSVString>>&&);

    void getHandleNames(CompletionHandler<void(ExceptionOr<Vector<String>>&&)>&&);
    void getHandle(const String& name, CompletionHandler<void(ExceptionOr<Ref<FileSystemHandle>>&&)>&&);

    class Iterator : public RefCounted<FileSystemDirectoryHandle::Iterator> {
    public:
        static Ref<Iterator> create(FileSystemDirectoryHandle&);
        using Result = std::optional<KeyValuePair<String, Ref<FileSystemHandle>>>;
        void next(CompletionHandler<void(ExceptionOr<Result>&&)>&&);
    private:
        explicit Iterator(FileSystemDirectoryHandle& source)
            : m_source(source)
        {
        }
        void advance(CompletionHandler<void(ExceptionOr<Result>&&)>&&);

        Ref<FileSystemDirectoryHandle> m_source;
        size_t m_index { 0 };
        Vector<String> m_keys;
        bool m_isInitialized { false };
        bool m_isWaitingForResult { false };
    };
    Ref<Iterator> createIterator(ScriptExecutionContext*);

private:
    FileSystemDirectoryHandle(ScriptExecutionContext&, String&&, FileSystemHandleIdentifier, Ref<FileSystemStorageConnection>&&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::FileSystemDirectoryHandle)
    static bool isType(const WebCore::FileSystemHandle& handle) { return handle.kind() == WebCore::FileSystemHandle::Kind::Directory; }
SPECIALIZE_TYPE_TRAITS_END()
