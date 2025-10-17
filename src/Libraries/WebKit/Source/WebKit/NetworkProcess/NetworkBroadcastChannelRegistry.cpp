/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include "NetworkBroadcastChannelRegistry.h"

#include "Logging.h"
#include "NetworkProcess.h"
#include "NetworkProcessProxyMessages.h"
#include "WebBroadcastChannelRegistryMessages.h"
#include <WebCore/MessageWithMessagePorts.h>
#include <wtf/CallbackAggregator.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

#define MESSAGE_CHECK(assertion, connection) MESSAGE_CHECK_BASE(assertion, connection)
#define MESSAGE_CHECK_COMPLETION(assertion, connection, completion) MESSAGE_CHECK_COMPLETION_BASE(assertion, connection, completion)

static bool isValidClientOrigin(const WebCore::ClientOrigin& clientOrigin)
{
    return !clientOrigin.topOrigin.isNull() && !clientOrigin.clientOrigin.isNull();
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkBroadcastChannelRegistry);

Ref<NetworkBroadcastChannelRegistry> NetworkBroadcastChannelRegistry::create(NetworkProcess& networkProcess)
{
    return adoptRef(*new NetworkBroadcastChannelRegistry(networkProcess));
}

NetworkBroadcastChannelRegistry::NetworkBroadcastChannelRegistry(NetworkProcess& networkProcess)
    : m_networkProcess(networkProcess)
{
}

NetworkBroadcastChannelRegistry::~NetworkBroadcastChannelRegistry() = default;

void NetworkBroadcastChannelRegistry::registerChannel(IPC::Connection& connection, const WebCore::ClientOrigin& origin, const String& name)
{
    MESSAGE_CHECK(isValidClientOrigin(origin), connection);

    auto& channelsForOrigin = m_broadcastChannels.ensure(origin, [] { return NameToConnectionIdentifiersMap { }; }).iterator->value;
    auto& connectionIdentifiersForName = channelsForOrigin.ensure(name, [] { return Vector<IPC::Connection::UniqueID> { }; }).iterator->value;
    ASSERT(!connectionIdentifiersForName.contains(connection.uniqueID()));
    connectionIdentifiersForName.append(connection.uniqueID());
}

void NetworkBroadcastChannelRegistry::unregisterChannel(IPC::Connection& connection, const WebCore::ClientOrigin& origin, const String& name)
{
    MESSAGE_CHECK(isValidClientOrigin(origin), connection);

    auto channelsForOriginIterator = m_broadcastChannels.find(origin);
    ASSERT(channelsForOriginIterator != m_broadcastChannels.end());
    if (channelsForOriginIterator == m_broadcastChannels.end())
        return;
    auto connectionIdentifiersForNameIterator = channelsForOriginIterator->value.find(name);
    ASSERT(connectionIdentifiersForNameIterator != channelsForOriginIterator->value.end());
    if (connectionIdentifiersForNameIterator == channelsForOriginIterator->value.end())
        return;

    ASSERT(connectionIdentifiersForNameIterator->value.contains(connection.uniqueID()));
    connectionIdentifiersForNameIterator->value.removeFirst(connection.uniqueID());
}

void NetworkBroadcastChannelRegistry::postMessage(IPC::Connection& connection, const WebCore::ClientOrigin& origin, const String& name, WebCore::MessageWithMessagePorts&& message, CompletionHandler<void()>&& completionHandler)
{
    MESSAGE_CHECK_COMPLETION(isValidClientOrigin(origin), connection, completionHandler());

    auto channelsForOriginIterator = m_broadcastChannels.find(origin);
    ASSERT(channelsForOriginIterator != m_broadcastChannels.end());
    if (channelsForOriginIterator == m_broadcastChannels.end())
        return completionHandler();
    auto connectionIdentifiersForNameIterator = channelsForOriginIterator->value.find(name);
    ASSERT(connectionIdentifiersForNameIterator != channelsForOriginIterator->value.end());
    if (connectionIdentifiersForNameIterator == channelsForOriginIterator->value.end())
        return completionHandler();

    auto callbackAggregator = CallbackAggregator::create(WTFMove(completionHandler));
    for (auto& connectionID : connectionIdentifiersForNameIterator->value) {
        // Only dispatch the post the messages to BroadcastChannels outside the source process.
        if (connectionID == connection.uniqueID())
            continue;

        RefPtr connection = IPC::Connection::connection(connectionID);
        if (!connection)
            continue;

        connection->sendWithAsyncReply(Messages::WebBroadcastChannelRegistry::PostMessageToRemote(origin, name, message), [callbackAggregator] { }, 0);
    }
}

void NetworkBroadcastChannelRegistry::removeConnection(IPC::Connection& connection)
{
    Vector<WebCore::ClientOrigin> originsToRemove;
    for (auto& entry : m_broadcastChannels) {
        Vector<String> namesToRemove;
        for (auto& innerEntry : entry.value) {
            innerEntry.value.removeFirst(connection.uniqueID());
            if (innerEntry.value.isEmpty())
                namesToRemove.append(innerEntry.key);
        }
        for (auto& nameToRemove : namesToRemove)
            entry.value.remove(nameToRemove);
        if (entry.value.isEmpty())
            originsToRemove.append(entry.key);
    }
    for (auto& originToRemove : originsToRemove)
        m_broadcastChannels.remove(originToRemove);
}

#undef MESSAGE_CHECK
#undef MESSAGE_CHECK_COMPLETION

} // namespace WebKit
