/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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


#if ENABLE(IPC_TESTING_API)

#include "IPCConnectionTesterIdentifier.h"
#include "IPCStreamTesterIdentifier.h"
#include "MessageReceiver.h"
#include "ScopedActiveMessageReceiveQueue.h"
#include "StreamConnectionBuffer.h"
#include "StreamConnectionWorkQueue.h"
#include "StreamMessageReceiver.h"
#include "StreamServerConnection.h"
#include <WebCore/SharedMemory.h>
#include <atomic>
#include <wtf/HashMap.h>
#include <wtf/WorkQueue.h>

#endif

namespace WebKit {

#if ENABLE(IPC_TESTING_API)

class IPCConnectionTester;
class IPCStreamTester;

// Main test interface for initiating various IPC test activities.
class IPCTester final : public IPC::MessageReceiver, public RefCounted<IPCTester> {
public:
    static Ref<IPCTester> create();
    ~IPCTester();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    // IPC::MessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;
private:
    IPCTester();

    // Messages.
    void startMessageTesting(IPC::Connection&, String&& driverName);
    void stopMessageTesting(CompletionHandler<void()>&&);
    void createStreamTester(IPC::Connection&, IPCStreamTesterIdentifier, IPC::StreamServerConnection::Handle&&);
    void releaseStreamTester(IPCStreamTesterIdentifier, CompletionHandler<void()>&&);
    void createConnectionTester(IPC::Connection&, IPCConnectionTesterIdentifier, IPC::Connection::Handle&&);
    void createConnectionTesterAndSendAsyncMessages(IPC::Connection&, IPCConnectionTesterIdentifier, IPC::Connection::Handle&&, uint32_t messageCount);
    void releaseConnectionTester(IPCConnectionTesterIdentifier, CompletionHandler<void()>&&);
    void sendSameSemaphoreBack(IPC::Connection&, IPC::Semaphore&&);
    void sendSemaphoreBackAndSignalProtocol(IPC::Connection&, IPC::Semaphore&&);
    void sendAsyncMessageToReceiver(IPC::Connection&, uint32_t);
    void asyncPing(uint32_t value, CompletionHandler<void(uint32_t)>&&);
    void syncPing(IPC::Connection&, uint32_t value, CompletionHandler<void(uint32_t)>&&);
    void syncPingEmptyReply(IPC::Connection&, uint32_t value, CompletionHandler<void()>&&);
    void asyncOptionalExceptionData(IPC::Connection&, bool sendEngaged, CompletionHandler<void(std::optional<WebCore::ExceptionData>, String)>&&);

    void stopIfNeeded();

    RefPtr<WorkQueue> m_testQueue;
    std::atomic<bool> m_shouldStop { false };

    using StreamTesterMap = HashMap<IPCStreamTesterIdentifier, IPC::ScopedActiveMessageReceiveQueue<IPCStreamTester>>;
    StreamTesterMap m_streamTesters;

    using ConnectionTesterMap = HashMap<IPCConnectionTesterIdentifier, IPC::ScopedActiveMessageReceiveQueue<IPCConnectionTester>>;
    ConnectionTesterMap m_connectionTesters;
};

#endif

}
