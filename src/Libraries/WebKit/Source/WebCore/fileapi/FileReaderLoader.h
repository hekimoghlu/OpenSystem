/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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

#include "BlobResourceHandle.h"
#include "ExceptionCode.h"
#include "ThreadableLoaderClient.h"
#include "URLKeepingBlobAlive.h"
#include <pal/text/TextEncoding.h>
#include <wtf/Forward.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {

class Blob;
class FileReaderLoaderClient;
class ScriptExecutionContext;
class TextResourceDecoder;
class ThreadableLoader;

class FileReaderLoader final : public ThreadableLoaderClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(FileReaderLoader);
public:
    enum ReadType {
        ReadAsArrayBuffer,
        ReadAsBinaryString,
        ReadAsBlob,
        ReadAsText,
        ReadAsDataURL,
        ReadAsBinaryChunks
    };

    // If client is given, do the loading asynchronously. Otherwise, load synchronously.
    WEBCORE_EXPORT FileReaderLoader(ReadType, FileReaderLoaderClient*);
    ~FileReaderLoader();

    WEBCORE_EXPORT void start(ScriptExecutionContext*, Blob&);
    void start(ScriptExecutionContext*, const URL&);
    WEBCORE_EXPORT void cancel();

    // ThreadableLoaderClient
    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) override;
    void didReceiveData(const SharedBuffer&) override;
    void didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&) override;
    void didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&) override;

    String stringResult();
    WEBCORE_EXPORT RefPtr<JSC::ArrayBuffer> arrayBufferResult() const;
    unsigned bytesLoaded() const { return m_bytesLoaded; }
    unsigned totalBytes() const { return m_totalBytes; }
    std::optional<ExceptionCode> errorCode() const { return m_errorCode; }

    void setEncoding(StringView);
    void setDataType(const String& dataType) { m_dataType = dataType; }

    const URL& url() { return m_urlForReading; }

    bool isCompleted() const;

private:
    void terminate();
    void cleanup();
    void failed(ExceptionCode);
    void convertToText();
    void convertToDataURL();
    bool processResponse(const ResourceResponse&);

    static ExceptionCode httpStatusCodeToErrorCode(int);
    static ExceptionCode toErrorCode(BlobResourceHandle::Error);

    ReadType m_readType;
    WeakPtr<FileReaderLoaderClient> m_client;
    PAL::TextEncoding m_encoding;
    String m_dataType;

    URLKeepingBlobAlive m_urlForReading;
    RefPtr<ThreadableLoader> m_loader;

    RefPtr<JSC::ArrayBuffer> m_rawData;
    bool m_isRawDataConverted;

    String m_stringResult;
    RefPtr<Blob> m_blobResult;

    // The decoder used to decode the text data.
    RefPtr<TextResourceDecoder> m_decoder;

    bool m_variableLength;
    unsigned m_bytesLoaded;
    unsigned m_totalBytes;

    std::optional<ExceptionCode> m_errorCode;
};

} // namespace WebCore
