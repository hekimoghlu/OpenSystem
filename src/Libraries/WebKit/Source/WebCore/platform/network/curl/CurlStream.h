/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#include "CurlContext.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/UniqueArray.h>
#include <wtf/Vector.h>

namespace WebCore {

class CurlStreamScheduler;
class SharedBuffer;
class SocketStreamError;

using CurlStreamID = uint16_t;
const CurlStreamID invalidCurlStreamID = 0;

class CurlStream {
    WTF_MAKE_TZONE_ALLOCATED(CurlStream);
    WTF_MAKE_NONCOPYABLE(CurlStream);
public:
    using LocalhostAlias = CurlHandle::LocalhostAlias;

    enum class ServerTrustEvaluation : bool { Disable, Enable };

    class Client {
    public:
        virtual void didOpen(CurlStreamID) = 0;
        virtual void didSendData(CurlStreamID, size_t) = 0;
        virtual void didReceiveData(CurlStreamID, const SharedBuffer&) = 0;
        virtual void didFail(CurlStreamID, CURLcode, CertificateInfo&&) = 0;
    };

    static std::unique_ptr<CurlStream> create(CurlStreamScheduler& scheduler, CurlStreamID streamID, URL&& url, ServerTrustEvaluation serverTrustEvaluation, LocalhostAlias localhostAlias)
    {
        return makeUnique<CurlStream>(scheduler, streamID, WTFMove(url), serverTrustEvaluation, localhostAlias);
    }

    CurlStream(CurlStreamScheduler&, CurlStreamID, URL&&, ServerTrustEvaluation, LocalhostAlias);
    virtual ~CurlStream();

    void send(UniqueArray<uint8_t>&&, size_t);

    void appendMonitoringFd(fd_set&, fd_set&, fd_set&, int&);
    void tryToTransfer(const fd_set&, const fd_set&, const fd_set&);

private:
    void destroyHandle();

    void tryToReceive();
    void tryToSend();

    void notifyFailure(CURLcode);

    static const size_t kReceiveBufferSize = 16 * 1024;

    CurlStreamScheduler& m_scheduler;
    CurlStreamID m_streamID;

    std::unique_ptr<CurlHandle> m_curlHandle;

    Vector<std::pair<UniqueArray<uint8_t>, size_t>> m_sendBuffers;
    size_t m_sendBufferOffset { 0 };
};

} // namespace WebCore
