/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#include "FileSystemWritableFileStream.h"

#include "InternalWritableStream.h"
#include "JSDOMPromise.h"
#include "JSDOMPromiseDeferred.h"
#include "JSFileSystemWritableFileStream.h"
#include <JavaScriptCore/JSPromise.h>

namespace WebCore {

ExceptionOr<Ref<FileSystemWritableFileStream>> FileSystemWritableFileStream::create(JSDOMGlobalObject& globalObject, Ref<WritableStreamSink>&& sink)
{
    auto result = createInternalWritableStream(globalObject, WTFMove(sink));
    if (result.hasException())
        return result.releaseException();

    return adoptRef(*new FileSystemWritableFileStream(result.releaseReturnValue()));
}

FileSystemWritableFileStream::FileSystemWritableFileStream(Ref<InternalWritableStream>&& internalStream)
    : WritableStream(WTFMove(internalStream))
{
}

static JSC::JSValue convertChunk(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const FileSystemWritableFileStream::ChunkType& data)
{
    return WTF::switchOn(data, [&](const RefPtr<JSC::ArrayBufferView>& arrayBufferView) {
        if (!arrayBufferView || arrayBufferView->isDetached())
            return JSC::jsNull();
        return toJS<IDLArrayBufferView>(lexicalGlobalObject, globalObject, *arrayBufferView);
    }, [&](const RefPtr<JSC::ArrayBuffer>& arrayBuffer) {
        if (!arrayBuffer || arrayBuffer->isDetached())
            return JSC::jsNull();
        return toJS<IDLArrayBuffer>(lexicalGlobalObject, globalObject, *arrayBuffer);
    }, [&](const RefPtr<Blob>& blob) {
        if (!blob)
            return JSC::jsNull();
        return toJS<IDLInterface<Blob>>(lexicalGlobalObject, globalObject, *blob);
    }, [&](const String& string) {
        return toJS<IDLDOMString>(lexicalGlobalObject, string);
    }, [&](const FileSystemWritableFileStream::WriteParams& params) {
        return toJS<IDLDictionary<FileSystemWritableFileStream::WriteParams>>(lexicalGlobalObject, globalObject, params);
    });
}

void FileSystemWritableFileStream::write(JSC::JSGlobalObject& lexicalGlobalObject, const ChunkType& data, DOMPromiseDeferred<void>&& promise)
{
    auto* globalObject = JSC::jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject);
    RELEASE_ASSERT(globalObject);

    auto jsData = convertChunk(lexicalGlobalObject, *globalObject, data);
    if (jsData == JSC::jsNull())
        return promise.reject(Exception { ExceptionCode::TypeError });

    Ref internalStream = internalWritableStream();
    auto result = internalStream->writeChunkForBingings(lexicalGlobalObject, jsData);
    if (result.hasException())
        return promise.reject(result.releaseException());

    auto* jsPromise = jsCast<JSC::JSPromise*>(result.returnValue());
    if (!jsPromise)
        return promise.reject(Exception { ExceptionCode::UnknownError, "Failed to complete write operation"_s });

    Ref domPromise = DOMPromise::create(*globalObject, *jsPromise);
    domPromise->whenSettled([domPromise, promise = WTFMove(promise)]() mutable {
        switch (domPromise->status()) {
        case DOMPromise::Status::Fulfilled:
            return promise.resolve();
        case DOMPromise::Status::Rejected:
            return promise.rejectWithCallback([&](auto&) {
                return domPromise->result();
            });
        case DOMPromise::Status::Pending:
            RELEASE_ASSERT_NOT_REACHED();
        }
    });
}

void FileSystemWritableFileStream::seek(JSC::JSGlobalObject& lexicalGlobalObject, uint64_t position, DOMPromiseDeferred<void>&& promise)
{
    WriteParams params { WriteCommandType::Seek, std::nullopt, position, std::nullopt };
    write(lexicalGlobalObject, params, WTFMove(promise));
}

void FileSystemWritableFileStream::truncate(JSC::JSGlobalObject& lexicalGlobalObject, uint64_t size, DOMPromiseDeferred<void>&& promise)
{
    WriteParams params { WriteCommandType::Truncate, size, std::nullopt, std::nullopt };
    write(lexicalGlobalObject, params, WTFMove(promise));
}

} // namespace WebCore
