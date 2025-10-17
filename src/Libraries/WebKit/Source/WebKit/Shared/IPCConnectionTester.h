/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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

#include "Connection.h"
#include "IPCConnectionTesterIdentifier.h"
#include "MessageReceiver.h"
#include <optional>
#include <wtf/HashMap.h>

namespace IPC {
class Connection;
}

namespace WebKit {

// Interface to test various IPC::Connection related activities.
class IPCConnectionTester final : public RefCounted<IPCConnectionTester>, private IPC::Connection::Client {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IPCConnectionTester);
public:
    ~IPCConnectionTester();
    static Ref<IPCConnectionTester> create(IPC::Connection&, IPCConnectionTesterIdentifier, IPC::Connection::Handle&&);
    void stopListeningForIPC(Ref<IPCConnectionTester>&& refFromConnection);

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void sendAsyncMessages(uint32_t messageCount);
private:
    IPCConnectionTester(Ref<IPC::Connection>&&, IPCConnectionTesterIdentifier, IPC::Connection::Handle&&);
    void initialize();

    // IPC::Connection::Client overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;
    void didClose(IPC::Connection&) final;
    void didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t) final;

    // Messages.
    void asyncMessage(uint32_t value);
    void syncMessage(uint32_t value, CompletionHandler<void(uint32_t sameValue)>&&);

    const Ref<IPC::Connection> m_connection;
    const Ref<IPC::Connection> m_testedConnection;
    const IPCConnectionTesterIdentifier m_identifier;
    uint32_t m_previousAsyncMessageValue { 0 };
};

}

#endif
