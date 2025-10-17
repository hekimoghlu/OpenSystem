/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

#include "IPCSemaphore.h"
#include "StreamServerConnection.h"
#include <atomic>
#include <wtf/Deque.h>
#include <wtf/FunctionDispatcher.h>
#include <wtf/Lock.h>
#include <wtf/Threading.h>
#include <wtf/Vector.h>

namespace IPC {

class WTF_CAPABILITY("is current") StreamConnectionWorkQueue final : public SerialFunctionDispatcher {
public:
    static Ref<StreamConnectionWorkQueue> create(ASCIILiteral name)
    {
        return adoptRef(*new StreamConnectionWorkQueue(name));
    }

    StreamConnectionWorkQueue(ASCIILiteral);
    ~StreamConnectionWorkQueue();
    void addStreamConnection(StreamServerConnection&);
    void removeStreamConnection(StreamServerConnection&);
    void stopAndWaitForCompletion(WTF::Function<void()>&& cleanupFunction = nullptr);
    void wakeUp();
    Semaphore& wakeUpSemaphore() { return m_wakeUpSemaphore; }

    // SerialFunctionDispatcher
    void dispatch(WTF::Function<void()>&&) final;
    bool isCurrent() const final;

private:
    void startProcessingThread() WTF_REQUIRES_LOCK(m_lock);
    void processStreams();

    ASCIILiteral m_name;

    Semaphore m_wakeUpSemaphore;
    std::atomic<bool> m_shouldQuit { false };

    mutable Lock m_lock;
    RefPtr<Thread> m_processingThread WTF_GUARDED_BY_LOCK(m_lock);
    Deque<Function<void()>> m_functions WTF_GUARDED_BY_LOCK(m_lock);
    WTF::Function<void()> m_cleanupFunction WTF_GUARDED_BY_LOCK(m_lock);
    Vector<Ref<StreamServerConnection>> m_connections WTF_GUARDED_BY_LOCK(m_lock);
};

}
