/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#include "MessagePortChannelRegistry.h"

#include "Logging.h"
#include <wtf/CompletionHandler.h>
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MessagePortChannelRegistry);

MessagePortChannelRegistry::MessagePortChannelRegistry() = default;

MessagePortChannelRegistry::~MessagePortChannelRegistry()
{
    ASSERT(m_openChannels.isEmpty());
}

void MessagePortChannelRegistry::didCreateMessagePortChannel(const MessagePortIdentifier& port1, const MessagePortIdentifier& port2)
{
    LOG(MessagePorts, "Registry: Creating MessagePortChannel %p linking %s and %s", this, port1.logString().utf8().data(), port2.logString().utf8().data());
    ASSERT(isMainThread());

    MessagePortChannel::create(*this, port1, port2);
}

void MessagePortChannelRegistry::messagePortChannelCreated(MessagePortChannel& channel)
{
    ASSERT(isMainThread());

    auto result = m_openChannels.add(channel.port1(), channel);
    ASSERT_UNUSED(result, result.isNewEntry);

    result = m_openChannels.add(channel.port2(), channel);
    ASSERT_UNUSED(result, result.isNewEntry);
}

void MessagePortChannelRegistry::messagePortChannelDestroyed(MessagePortChannel& channel)
{
    ASSERT(isMainThread());

    ASSERT(m_openChannels.get(channel.port1()) == &channel);
    ASSERT(m_openChannels.get(channel.port2()) == &channel);

    m_openChannels.remove(channel.port1());
    m_openChannels.remove(channel.port2());

    LOG(MessagePorts, "Registry: After removing channel %s there are %u channels left in the registry:", channel.logString().utf8().data(), m_openChannels.size());
}

void MessagePortChannelRegistry::didEntangleLocalToRemote(const MessagePortIdentifier& local, const MessagePortIdentifier& remote, ProcessIdentifier process)
{
    ASSERT(isMainThread());

    // The channel might be gone if the remote side was closed.
    RefPtr channel = m_openChannels.get(local);
    if (!channel)
        return;

    ASSERT_UNUSED(remote, channel->includesPort(remote));

    channel->entanglePortWithProcess(local, process);
}

void MessagePortChannelRegistry::didDisentangleMessagePort(const MessagePortIdentifier& port)
{
    ASSERT(isMainThread());

    // The channel might be gone if the remote side was closed.
    if (RefPtr channel = m_openChannels.get(port))
        channel->disentanglePort(port);
}

void MessagePortChannelRegistry::didCloseMessagePort(const MessagePortIdentifier& port)
{
    ASSERT(isMainThread());

    LOG(MessagePorts, "Registry: MessagePort %s closed in registry", port.logString().utf8().data());

    RefPtr channel = m_openChannels.get(port);
    if (!channel)
        return;

#ifndef NDEBUG
    if (channel && channel->hasAnyMessagesPendingOrInFlight())
        LOG(MessagePorts, "Registry: (Note) The channel closed for port %s had messages pending or in flight", port.logString().utf8().data());
#endif

    channel->closePort(port);

    // FIXME: When making message ports be multi-process, this should probably push a notification
    // to the remaining port to tell it this port closed.
}

bool MessagePortChannelRegistry::didPostMessageToRemote(MessageWithMessagePorts&& message, const MessagePortIdentifier& remoteTarget)
{
    ASSERT(isMainThread());

    LOG(MessagePorts, "Registry: Posting message to MessagePort %s in registry", remoteTarget.logString().utf8().data());

    // The channel might be gone if the remote side was closed.
    RefPtr channel = m_openChannels.get(remoteTarget);
    if (!channel) {
        LOG(MessagePorts, "Registry: Could not find MessagePortChannel for port %s; It was probably closed. Message will be dropped.", remoteTarget.logString().utf8().data());
        return false;
    }

    return channel->postMessageToRemote(WTFMove(message), remoteTarget);
}

void MessagePortChannelRegistry::takeAllMessagesForPort(const MessagePortIdentifier& port, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&& callback)
{
    ASSERT(isMainThread());

    LOG(MessagePorts, "Registry: Taking all messages for MessagePort %s", port.logString().utf8().data());

    // The channel might be gone if the remote side was closed.
    RefPtr channel = m_openChannels.get(port);
    if (!channel) {
        callback({ }, [] { });
        return;
    }

    channel->takeAllMessagesForPort(port, WTFMove(callback));
}

MessagePortChannel* MessagePortChannelRegistry::existingChannelContainingPort(const MessagePortIdentifier& port)
{
    ASSERT(isMainThread());

    return m_openChannels.get(port);
}

} // namespace WebCore
