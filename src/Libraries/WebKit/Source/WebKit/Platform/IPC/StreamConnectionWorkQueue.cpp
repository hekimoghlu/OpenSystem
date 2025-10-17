/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include "StreamConnectionWorkQueue.h"

#if USE(FOUNDATION)
#include <wtf/AutodrainedPool.h>
#endif

namespace IPC {

StreamConnectionWorkQueue::StreamConnectionWorkQueue(ASCIILiteral name)
    : m_name(name)
{
}

StreamConnectionWorkQueue::~StreamConnectionWorkQueue()
{
    // `StreamConnectionWorkQueue::stopAndWaitForCompletion()` should be called if anything has been dispatched or listened to.
    ASSERT(!m_processingThread);
}

void StreamConnectionWorkQueue::dispatch(WTF::Function<void()>&& function)
{
    {
        Locker locker { m_lock };
        // Currently stream IPC::Connection::Client::didClose is delivered after
        // scheduling the work queue stop. This is ignored, as the client function
        // is not used at the moment. Later on, the closure signal should be set synchronously,
        // not dispatched as a function.
        if (m_shouldQuit)
            return;
        m_functions.append(WTFMove(function));
        if (!m_shouldQuit && !m_processingThread) {
            startProcessingThread();
            return;
        }
    }
    wakeUp();
}

void StreamConnectionWorkQueue::addStreamConnection(StreamServerConnection& connection)
{
    {
        Locker locker { m_lock };
        ASSERT(m_connections.findIf([&connection](StreamServerConnection& other) { return &other == &connection; }) == notFound); // NOLINT
        m_connections.append(connection);
        if (!m_shouldQuit && !m_processingThread) {
            startProcessingThread();
            return;
        }
    }
    wakeUp();
}

void StreamConnectionWorkQueue::removeStreamConnection(StreamServerConnection& connection)
{
    {
        Locker locker { m_lock };
        bool didRemove = m_connections.removeFirstMatching([&connection](StreamServerConnection& other) {
            return &other == &connection;
        });
        ASSERT_UNUSED(didRemove, didRemove);
    }
    wakeUp();
}

void StreamConnectionWorkQueue::stopAndWaitForCompletion(WTF::Function<void()>&& cleanupFunction)
{
    RefPtr<Thread> processingThread;
    {
        Locker locker { m_lock };
        m_cleanupFunction = WTFMove(cleanupFunction);
        processingThread = m_processingThread;
        m_shouldQuit = true;
    }
    if (!processingThread)
        return;
    ASSERT(Thread::current().uid() != processingThread->uid());
    wakeUp();
    processingThread->waitForCompletion();
    {
        Locker locker { m_lock };
        m_processingThread = nullptr;
    }
}

void StreamConnectionWorkQueue::wakeUp()
{
    m_wakeUpSemaphore.signal();
}

void StreamConnectionWorkQueue::startProcessingThread()
{
    auto task = [this]() mutable {
        for (;;) {
            processStreams();
            if (m_shouldQuit) {
                processStreams();
                WTF::Function<void()> cleanup = nullptr;
                {
                    Locker locker { m_lock };
                    cleanup = WTFMove(m_cleanupFunction);

                }
                if (cleanup)
                    cleanup();
                return;
            }
            m_wakeUpSemaphore.wait();
        }
    };
    m_processingThread = Thread::create(m_name, WTFMove(task), ThreadType::Graphics, Thread::QOS::UserInteractive);
}

void StreamConnectionWorkQueue::processStreams()
{
    constexpr size_t defaultMessageLimit = 1000;
    bool hasMoreToProcess = false;
    do {
#if USE(FOUNDATION)
        AutodrainedPool perProcessingIterationPool;
#endif
        Deque<WTF::Function<void()>> functions;
        Vector<Ref<StreamServerConnection>> connections;
        {
            Locker locker { m_lock };
            functions.swap(m_functions);
            connections = m_connections;
        }
        for (auto& function : functions)
            WTFMove(function)();

        hasMoreToProcess = false;
        for (auto& connection : connections)
            hasMoreToProcess |= connection->dispatchStreamMessages(defaultMessageLimit) == StreamServerConnection::HasMoreMessages;
    } while (hasMoreToProcess);
}

bool StreamConnectionWorkQueue::isCurrent() const
{
    Locker locker { m_lock };
    return m_processingThread ? m_processingThread->uid() == Thread::current().uid() : false;
}

}
