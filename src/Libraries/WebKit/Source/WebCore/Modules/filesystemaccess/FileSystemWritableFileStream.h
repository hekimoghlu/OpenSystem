/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#include "FileSystemWriteCommandType.h"
#include "WritableStream.h"

namespace WebCore {

class Blob;
template<typename> class DOMPromiseDeferred;

class FileSystemWritableFileStream : public WritableStream {
public:
    static ExceptionOr<Ref<FileSystemWritableFileStream>> create(JSDOMGlobalObject&, Ref<WritableStreamSink>&&);

    using WriteCommandType = FileSystemWriteCommandType;
    using DataVariant = std::variant<RefPtr<JSC::ArrayBufferView>, RefPtr<JSC::ArrayBuffer>, RefPtr<Blob>, String>;
    struct WriteParams {
        WriteCommandType type;
        std::optional<uint64_t> size;
        std::optional<uint64_t> position;
        std::optional<DataVariant> data;
    };

    using ChunkType = std::variant<RefPtr<JSC::ArrayBufferView>, RefPtr<JSC::ArrayBuffer>, RefPtr<Blob>, String, WriteParams>;
    void write(JSC::JSGlobalObject&, const ChunkType&, DOMPromiseDeferred<void>&&);
    void seek(JSC::JSGlobalObject&, uint64_t position, DOMPromiseDeferred<void>&&);
    void truncate(JSC::JSGlobalObject&, uint64_t size, DOMPromiseDeferred<void>&&);
    WritableStream::Type type() const final { return WritableStream::Type::FileSystem; }

private:
    explicit FileSystemWritableFileStream(Ref<InternalWritableStream>&&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::FileSystemWritableFileStream)
    static bool isType(const WebCore::WritableStream& stream) { return stream.type() == WebCore::WritableStream::Type::FileSystem; }
SPECIALIZE_TYPE_TRAITS_END()
