/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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

#include "FileStreamClient.h"
#include "ResourceHandle.h"
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class AsyncFileStream;
class BlobData;
class FileStream;
class ResourceHandleClient;
class ResourceRequest;
class BlobDataItem;

class BlobResourceHandle final : public FileStreamClient, public ResourceHandle  {
public:
    static Ref<BlobResourceHandle> createAsync(BlobData*, const ResourceRequest&, ResourceHandleClient*);

    static void loadResourceSynchronously(BlobData*, const ResourceRequest&, ResourceError&, ResourceResponse&, Vector<uint8_t>& data);

    void start();
    int readSync(std::span<uint8_t>);

    bool aborted() const { return m_aborted; }

    enum class Error {
        NoError = 0,
        NotFoundError = 1,
        SecurityError = 2,
        RangeError = 3,
        NotReadableError = 4,
        MethodNotAllowed = 5
    };

private:
    BlobResourceHandle(BlobData*, const ResourceRequest&, ResourceHandleClient*, bool async);
    virtual ~BlobResourceHandle();

    // FileStreamClient methods.
    void didGetSize(long long) override;
    void didOpen(bool) override;
    void didRead(int) override;

    // ResourceHandle methods.
    void cancel() override;

    void doStart();
    void getSizeForNext();
    std::optional<Error> seek();
    void consumeData(std::span<const uint8_t>);
    void failed(Error);

    void readAsync();
    void readDataAsync(const BlobDataItem&);
    void readFileAsync(const BlobDataItem&);

    int readDataSync(const BlobDataItem&, std::span<uint8_t>);
    int readFileSync(const BlobDataItem&, std::span<uint8_t>);

    void notifyResponse();
    void notifyResponseOnSuccess();
    void notifyResponseOnError();
    void notifyReceiveData(std::span<const uint8_t>);
    void notifyFail(Error);
    void notifyFinish();

    bool erroredOrAborted() const { return m_aborted || m_errorCode != Error::NoError; }

    enum { kPositionNotSpecified = -1 };

    RefPtr<BlobData> m_blobData;
    bool m_async;
    std::unique_ptr<AsyncFileStream> m_asyncStream; // For asynchronous loading.
    std::unique_ptr<FileStream> m_stream; // For synchronous loading.
    Vector<uint8_t> m_buffer;
    Vector<long long> m_itemLengthList;
    Error m_errorCode { Error::NoError };
    bool m_aborted { false };
    bool m_isRangeRequest { false };
    long long m_rangeStart { kPositionNotSpecified };
    long long m_rangeEnd { kPositionNotSpecified };
    long long m_totalSize { 0 };
    long long m_totalRemainingSize { 0 };
    long long m_currentItemReadSize { 0 };
    unsigned m_sizeItemCount { 0 };
    unsigned m_readItemCount { 0 };
    bool m_fileOpened { false };
};

} // namespace WebCore
