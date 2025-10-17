/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
#include "CurlStreamScheduler.h"
#include <wtf/TZoneMallocInlines.h>

#if USE(CURL)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CurlStreamScheduler);

CurlStreamScheduler::CurlStreamScheduler()
{
    ASSERT(isMainThread());
}

CurlStreamScheduler::~CurlStreamScheduler()
{
    ASSERT(isMainThread());
}

CurlStreamID CurlStreamScheduler::createStream(const URL& url, CurlStream::Client& client, CurlStream::ServerTrustEvaluation serverTrustEvaluation, CurlStream::LocalhostAlias localhostAlias)
{
    ASSERT(isMainThread());

    do {
        m_currentStreamID = (m_currentStreamID + 1 != invalidCurlStreamID) ? m_currentStreamID + 1 : 1;
    } while (m_clientList.contains(m_currentStreamID));

    auto streamID = m_currentStreamID;
    m_clientList.add(streamID, &client);

    callOnWorkerThread([this, streamID, url = url.isolatedCopy(), serverTrustEvaluation, localhostAlias]() mutable {
        m_streamList.add(streamID, CurlStream::create(*this, streamID, WTFMove(url), serverTrustEvaluation, localhostAlias));
    });

    return streamID;
}

void CurlStreamScheduler::destroyStream(CurlStreamID streamID)
{
    ASSERT(isMainThread());

    if (m_clientList.contains(streamID))
        m_clientList.remove(streamID);

    callOnWorkerThread([this, streamID]() {
        if (m_streamList.contains(streamID))
            m_streamList.remove(streamID);
    });
}

void CurlStreamScheduler::send(CurlStreamID streamID, UniqueArray<uint8_t>&& data, size_t length)
{
    ASSERT(isMainThread());

    callOnWorkerThread([this, streamID, data = WTFMove(data), length]() mutable {
        if (auto stream = m_streamList.get(streamID))
            stream->send(WTFMove(data), length);
    });
}

void CurlStreamScheduler::callOnWorkerThread(Function<void()>&& task)
{
    ASSERT(isMainThread());

    {
        Locker locker { m_mutex };
        m_taskQueue.append(WTFMove(task));
    }

    startThreadIfNeeded();
}

void CurlStreamScheduler::callClientOnMainThread(CurlStreamID streamID, Function<void(CurlStream::Client&)>&& task)
{
    ASSERT(!isMainThread());

    callOnMainThread([this, streamID, task = WTFMove(task)]() {
        if (auto client = m_clientList.get(streamID))
            task(*client);
    });
}

void CurlStreamScheduler::startThreadIfNeeded()
{
    {
        Locker locker { m_mutex };
        if (m_runThread)
            return;
    }

    if (m_thread)
        m_thread->waitForCompletion();

    m_runThread = true;

    m_thread = Thread::create("curlStreamThread"_s, [this] {
        workerThread();
    }, ThreadType::Network);
}

void CurlStreamScheduler::stopThreadIfNoMoreJobRunning()
{
    ASSERT(!isMainThread());

    if (m_streamList.size())
        return;

    Locker locker { m_mutex };
    if (m_taskQueue.size())
        return;

    m_runThread = false;
}

void CurlStreamScheduler::executeTasks()
{
    ASSERT(!isMainThread());

    Vector<Function<void()>> taskQueue;

    {
        Locker locker { m_mutex };
        taskQueue = WTFMove(m_taskQueue);
    }

    for (auto& task : taskQueue)
        task();
}

void CurlStreamScheduler::workerThread()
{
    ASSERT(!isMainThread());
    static const int selectTimeoutMS = 20;
    struct timeval timeout { 0, selectTimeoutMS * 1000};

    while (m_runThread) {
        executeTasks();

        int rc = 0;
        fd_set readfds;
        fd_set writefds;
        fd_set exceptfds;

        do {
            int maxfd = -1;

            FD_ZERO(&readfds);
            FD_ZERO(&writefds);
            FD_ZERO(&exceptfds);

            for (auto& stream : m_streamList.values())
                stream->appendMonitoringFd(readfds, writefds, exceptfds, maxfd);

            if (maxfd >= 0)
                rc = ::select(maxfd + 1, &readfds, &writefds, &exceptfds, &timeout);
        } while (rc == -1 && errno == EINTR);

        for (auto& stream : m_streamList.values())
            stream->tryToTransfer(readfds, writefds, exceptfds);

        stopThreadIfNoMoreJobRunning();
    }
}

}

#endif
