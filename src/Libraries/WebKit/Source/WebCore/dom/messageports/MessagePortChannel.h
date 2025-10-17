/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

#include "MessagePortChannelProvider.h"
#include "MessagePortIdentifier.h"
#include "MessageWithMessagePorts.h"
#include "ProcessIdentifier.h"
#include <wtf/HashSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class MessagePortChannelRegistry;

class MessagePortChannel : public RefCountedAndCanMakeWeakPtr<MessagePortChannel> {
public:
    static Ref<MessagePortChannel> create(MessagePortChannelRegistry&, const MessagePortIdentifier& port1, const MessagePortIdentifier& port2);

    WEBCORE_EXPORT ~MessagePortChannel();

    const MessagePortIdentifier& port1() const { return m_ports[0]; }
    const MessagePortIdentifier& port2() const { return m_ports[1]; }

    WEBCORE_EXPORT std::optional<ProcessIdentifier> processForPort(const MessagePortIdentifier&);
    bool includesPort(const MessagePortIdentifier&);
    void entanglePortWithProcess(const MessagePortIdentifier&, ProcessIdentifier);
    void disentanglePort(const MessagePortIdentifier&);
    void closePort(const MessagePortIdentifier&);
    bool postMessageToRemote(MessageWithMessagePorts&&, const MessagePortIdentifier& remoteTarget);

    void takeAllMessagesForPort(const MessagePortIdentifier&, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&&);

    WEBCORE_EXPORT bool hasAnyMessagesPendingOrInFlight() const;

    uint64_t beingTransferredCount();

#if !LOG_DISABLED
    String logString() const { return makeString(m_ports[0].logString(), ':', m_ports[1].logString()); }
#endif

private:
    MessagePortChannel(MessagePortChannelRegistry&, const MessagePortIdentifier& port1, const MessagePortIdentifier& port2);

    CheckedRef<MessagePortChannelRegistry> checkedRegistry() const;

    std::array<MessagePortIdentifier, 2> m_ports;
    std::array<bool, 2> m_isClosed { false, false };
    std::array<std::optional<ProcessIdentifier>, 2> m_processes;
    std::array<RefPtr<MessagePortChannel>, 2> m_entangledToProcessProtectors;
    std::array<Vector<MessageWithMessagePorts>, 2> m_pendingMessages;
    std::array<UncheckedKeyHashSet<RefPtr<MessagePortChannel>>, 2> m_pendingMessagePortTransfers;
    std::array<RefPtr<MessagePortChannel>, 2> m_pendingMessageProtectors;
    uint64_t m_messageBatchesInFlight { 0 };

    CheckedRef<MessagePortChannelRegistry> m_registry;
};

} // namespace WebCore
