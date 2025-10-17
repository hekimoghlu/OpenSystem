/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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

#include "NetworkDataTask.h"
#include <WebCore/FileStreamClient.h>
#include <wtf/FileSystem.h>

namespace WebCore {
class AsyncFileStream;
class BlobDataFileReference;
class BlobData;
class BlobDataItem;
}

namespace WebKit {

class NetworkProcess;

class NetworkDataTaskBlob final : public NetworkDataTask, public WebCore::FileStreamClient {
public:
    static Ref<NetworkDataTask> create(NetworkSession& session, NetworkDataTaskClient& client, const WebCore::ResourceRequest& request, const Vector<RefPtr<WebCore::BlobDataFileReference>>& fileReferences, const RefPtr<WebCore::SecurityOrigin>& topOrigin)
    {
        return adoptRef(*new NetworkDataTaskBlob(session, client, request, fileReferences, topOrigin));
    }

    ~NetworkDataTaskBlob();

private:
    NetworkDataTaskBlob(NetworkSession&, NetworkDataTaskClient&, const WebCore::ResourceRequest&, const Vector<RefPtr<WebCore::BlobDataFileReference>>&, const RefPtr<WebCore::SecurityOrigin>& topOrigin);

    void cancel() override;
    void resume() override;
    void invalidateAndCancel() override;
    NetworkDataTask::State state() const override { return m_state; }

    void setPendingDownloadLocation(const String&, SandboxExtension::Handle&&, bool /*allowOverwrite*/) override;
    String suggestedFilename() const override;

    // FileStreamClient methods.
    void didGetSize(long long) override;
    void didOpen(bool) override;
    void didRead(int) override;

    enum class Error {
        NoError = 0,
        NotFoundError = 1,
        SecurityError = 2,
        RangeError = 3,
        NotReadableError = 4,
        MethodNotAllowed = 5
    };

    void clearStream();
    void getSizeForNext();
    void dispatchDidReceiveResponse();
    std::optional<Error> seek();
    void consumeData(std::span<const uint8_t>);
    void read();
    void readData(const WebCore::BlobDataItem&);
    void readFile(const WebCore::BlobDataItem&);
    void download();
    bool writeDownload(std::span<const uint8_t>);
    void cleanDownloadFiles();
    void didFailDownload(const WebCore::ResourceError&);
    void didFinishDownload();
    void didFail(Error);
    void didFinish();

    enum { kPositionNotSpecified = -1 };

    RefPtr<WebCore::BlobData> m_blobData;
    std::unique_ptr<WebCore::AsyncFileStream> m_stream; // For asynchronous loading.
    Vector<uint8_t> m_buffer;
    Vector<long long> m_itemLengthList;
    State m_state { State::Suspended };
    bool m_isRangeRequest { false };
    long long m_rangeStart { kPositionNotSpecified };
    long long m_rangeEnd { kPositionNotSpecified };
    long long m_totalSize { 0 };
    long long m_downloadBytesWritten { 0 };
    long long m_totalRemainingSize { 0 };
    long long m_currentItemReadSize { 0 };
    unsigned m_sizeItemCount { 0 };
    unsigned m_readItemCount { 0 };
    bool m_fileOpened { false };
    FileSystem::PlatformFileHandle m_downloadFile { FileSystem::invalidPlatformFileHandle };

    Vector<RefPtr<WebCore::BlobDataFileReference>> m_fileReferences;
    RefPtr<SandboxExtension> m_sandboxExtension;
    Ref<NetworkProcess> m_networkProcess;
};

} // namespace WebKit
