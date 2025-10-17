/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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

#include "IPCStreamTesterIdentifier.h"
#include "ScopedActiveMessageReceiveQueue.h"
#include "StreamMessageReceiver.h"
#include "StreamServerConnection.h"
#include <WebCore/SharedMemory.h>
#include <memory>
#include <wtf/HashMap.h>

namespace IPC {
class Connection;
class StreamConnectionBuffer;
class StreamConnectionWorkQueue;
}

namespace WebKit {

// Interface to test various IPC stream related activities.
class IPCStreamTester final : public IPC::StreamMessageReceiver {
public:
    static RefPtr<IPCStreamTester> create(IPCStreamTesterIdentifier, IPC::StreamServerConnection::Handle&&, bool ignoreInvalidMessageForTesting);
    void stopListeningForIPC(Ref<IPCStreamTester>&& refFromConnection);

    // IPC::StreamMessageReceiver overrides.
    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;
private:
    IPCStreamTester(IPCStreamTesterIdentifier, IPC::StreamServerConnection::Handle&&, bool ignoreInvalidMessageForTesting);
    ~IPCStreamTester();
    void initialize();
    IPC::StreamConnectionWorkQueue& workQueue() const { return m_workQueue; }
    Ref<IPC::StreamConnectionWorkQueue> protectedWorkQueue() const { return m_workQueue; }

    // Messages.
    void syncMessage(uint32_t value, CompletionHandler<void(uint32_t)>&&);
    void syncMessageNotStreamEncodableReply(uint32_t value, CompletionHandler<void(uint32_t)>&&);
    void syncMessageNotStreamEncodableBoth(uint32_t value, CompletionHandler<void(uint32_t)>&&);
    void syncMessageReturningSharedMemory1(uint32_t byteCount, CompletionHandler<void(std::optional<WebCore::SharedMemory::Handle>&&)>&&);
    void syncMessageEmptyReply(uint32_t, CompletionHandler<void()>&&);
    void syncCrashOnZero(int32_t, CompletionHandler<void(int32_t)>&&);
    void checkAutoreleasePool(CompletionHandler<void(int32_t)>&&);
    void asyncPing(uint32_t value, CompletionHandler<void(uint32_t)>&&);
    void emptyMessage();

    const Ref<IPC::StreamConnectionWorkQueue> m_workQueue;
    const Ref<IPC::StreamServerConnection> m_streamConnection;
    const IPCStreamTesterIdentifier m_identifier;
    std::shared_ptr<bool> m_autoreleasePoolCheckValue;
};

}

#endif
