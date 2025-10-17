/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include "MessagePortChannel.h"

#include "Logging.h"
#include "MessagePortChannelRegistry.h"
#include <wtf/CompletionHandler.h>
#include <wtf/MainThread.h>

namespace WebCore {

Ref<MessagePortChannel> MessagePortChannel::create(MessagePortChannelRegistry& registry, const MessagePortIdentifier& port1, const MessagePortIdentifier& port2)
{
    return adoptRef(*new MessagePortChannel(registry, port1, port2));
}

MessagePortChannel::MessagePortChannel(MessagePortChannelRegistry& registry, const MessagePortIdentifier& port1, const MessagePortIdentifier& port2)
    : m_ports { port1, port2 }
    , m_registry(registry)
{
    ASSERT(isMainThread());

    relaxAdoptionRequirement();

    m_processes[0] = port1.processIdentifier;
    m_entangledToProcessProtectors[0] = this;
    m_processes[1] = port2.processIdentifier;
    m_entangledToProcessProtectors[1] = this;

    checkedRegistry()->messagePortChannelCreated(*this);
}

MessagePortChannel::~MessagePortChannel()
{
    checkedRegistry()->messagePortChannelDestroyed(*this);
}

CheckedRef<MessagePortChannelRegistry> MessagePortChannel::checkedRegistry() const
{
    return m_registry;
}

std::optional<ProcessIdentifier> MessagePortChannel::processForPort(const MessagePortIdentifier& port)
{
    ASSERT(isMainThread());
    ASSERT(port == m_ports[0] || port == m_ports[1]);
    size_t i = port == m_ports[0] ? 0 : 1;
    return m_processes[i];
}

bool MessagePortChannel::includesPort(const MessagePortIdentifier& port)
{
    ASSERT(isMainThread());

    return m_ports[0] == port || m_ports[1] == port;
}

void MessagePortChannel::entanglePortWithProcess(const MessagePortIdentifier& port, ProcessIdentifier process)
{
    ASSERT(isMainThread());

    ASSERT(port == m_ports[0] || port == m_ports[1]);
    size_t i = port == m_ports[0] ? 0 : 1;

    LOG(MessagePorts, "MessagePortChannel %s (%p) entangling port %s (that port has %zu messages available)", logString().utf8().data(), this, port.logString().utf8().data(), m_pendingMessages[i].size());

    ASSERT(!m_processes[i] || *m_processes[i] == process);
    m_processes[i] = process;
    m_entangledToProcessProtectors[i] = this;
    m_pendingMessagePortTransfers[i].remove(this);
}

void MessagePortChannel::disentanglePort(const MessagePortIdentifier& port)
{
    ASSERT(isMainThread());

    LOG(MessagePorts, "MessagePortChannel %s (%p) disentangling port %s", logString().utf8().data(), this, port.logString().utf8().data());

    ASSERT(port == m_ports[0] || port == m_ports[1]);
    size_t i = port == m_ports[0] ? 0 : 1;

    ASSERT(m_processes[i] || m_isClosed[i]);
    m_processes[i] = std::nullopt;
    m_pendingMessagePortTransfers[i].add(this);

    // This set of steps is to guarantee that the lock is unlocked before the
    // last ref to this object is released.
    auto protectedThis = WTFMove(m_entangledToProcessProtectors[i]);
}

void MessagePortChannel::closePort(const MessagePortIdentifier& port)
{
    ASSERT(isMainThread());

    ASSERT(port == m_ports[0] || port == m_ports[1]);
    size_t i = port == m_ports[0] ? 0 : 1;

    m_processes[i] = std::nullopt;
    m_isClosed[i] = true;

    m_pendingMessages[i].clear();
    m_pendingMessagePortTransfers[i].clear();
    m_pendingMessageProtectors[i] = nullptr;
    m_entangledToProcessProtectors[i] = nullptr;
}

bool MessagePortChannel::postMessageToRemote(MessageWithMessagePorts&& message, const MessagePortIdentifier& remoteTarget)
{
    ASSERT(isMainThread());

    ASSERT(remoteTarget == m_ports[0] || remoteTarget == m_ports[1]);
    size_t i = remoteTarget == m_ports[0] ? 0 : 1;

    m_pendingMessages[i].append(WTFMove(message));
    LOG(MessagePorts, "MessagePortChannel %s (%p) now has %zu messages pending on port %s", logString().utf8().data(), this, m_pendingMessages[i].size(), remoteTarget.logString().utf8().data());

    if (m_pendingMessages[i].size() == 1) {
        m_pendingMessageProtectors[i] = this;
        return true;
    }

    ASSERT(m_pendingMessageProtectors[i] == this);
    return false;
}

void MessagePortChannel::takeAllMessagesForPort(const MessagePortIdentifier& port, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&& callback)
{
    ASSERT(isMainThread());

    LOG(MessagePorts, "MessagePortChannel %p taking all messages for port %s", this, port.logString().utf8().data());

    ASSERT(port == m_ports[0] || port == m_ports[1]);
    size_t i = port == m_ports[0] ? 0 : 1;

    if (m_pendingMessages[i].isEmpty()) {
        callback({ }, [] { });
        return;
    }

    ASSERT(m_pendingMessageProtectors[i] == this);

    Vector<MessageWithMessagePorts> result;
    result.swap(m_pendingMessages[i]);

    ++m_messageBatchesInFlight;

    LOG(MessagePorts, "There are %zu messages to take for port %s. Taking them now, messages in flight is now %" PRIu64, result.size(), port.logString().utf8().data(), m_messageBatchesInFlight);

    auto size = result.size();
    callback(WTFMove(result), [size, port, protectedThis = WTFMove(m_pendingMessageProtectors[i])] {
        UNUSED_PARAM(port);
#if LOG_DISABLED
        UNUSED_PARAM(size);
#endif
        --(protectedThis->m_messageBatchesInFlight);
        LOG(MessagePorts, "Message port channel %s was notified that a batch of %zu message port messages targeted for port %s just completed dispatch, in flight is now %" PRIu64, protectedThis->logString().utf8().data(), size, port.logString().utf8().data(), protectedThis->m_messageBatchesInFlight);

    });
}

bool MessagePortChannel::hasAnyMessagesPendingOrInFlight() const
{
    ASSERT(isMainThread());
    return m_messageBatchesInFlight || !m_pendingMessages[0].isEmpty() || !m_pendingMessages[1].isEmpty();
}

} // namespace WebCore
