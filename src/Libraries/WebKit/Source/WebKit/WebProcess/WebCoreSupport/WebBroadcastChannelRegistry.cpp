/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#include "WebBroadcastChannelRegistry.h"

#include "NetworkBroadcastChannelRegistryMessages.h"
#include "NetworkProcessConnection.h"
#include "WebProcess.h"
#include <WebCore/BroadcastChannel.h>
#include <WebCore/MessageWithMessagePorts.h>
#include <wtf/CallbackAggregator.h>

namespace WebKit {

static inline IPC::Connection& networkProcessConnection()
{
    return WebProcess::singleton().ensureNetworkProcessConnection().connection();
}

// Opaque origins are only stored in process in m_channelsPerOrigin and never sent to the NetworkProcess as a ClientOrigin.
// The identity of opaque origins wouldn't be preserved when serializing them as a SecurityOriginData (via ClientOrigin).
// Since BroadcastChannels from an opaque origin can only communicate with other BroadcastChannels from the same opaque origin,
// the destination channels have to be within the same WebProcess anyway.
static std::optional<WebCore::ClientOrigin> toClientOrigin(const WebCore::PartitionedSecurityOrigin& origin)
{
    if (origin.topOrigin->isOpaque() || origin.clientOrigin->isOpaque())
        return std::nullopt;
    return WebCore::ClientOrigin { origin.topOrigin->data(), origin.clientOrigin->data() };
}

void WebBroadcastChannelRegistry::registerChannel(const WebCore::PartitionedSecurityOrigin& origin, const String& name, WebCore::BroadcastChannelIdentifier identifier)
{
    auto& channelsForOrigin = m_channelsPerOrigin.ensure(origin, [] {
        return HashMap<String, Vector<WebCore::BroadcastChannelIdentifier>> { };
    }).iterator->value;
    auto& channelsForName = channelsForOrigin.ensure(name, [] { return Vector<WebCore::BroadcastChannelIdentifier> { }; }).iterator->value;
    channelsForName.append(identifier);

    if (channelsForName.size() == 1) {
        if (auto clientOrigin = toClientOrigin(origin))
            networkProcessConnection().send(Messages::NetworkBroadcastChannelRegistry::RegisterChannel { *clientOrigin, name }, 0);
    }
}

void WebBroadcastChannelRegistry::unregisterChannel(const WebCore::PartitionedSecurityOrigin& origin, const String& name, WebCore::BroadcastChannelIdentifier identifier)
{
    auto channelsPerOriginIterator = m_channelsPerOrigin.find(origin);
    if (channelsPerOriginIterator == m_channelsPerOrigin.end())
        return;

    auto& channelsForOrigin = channelsPerOriginIterator->value;
    auto channelsForOriginIterator = channelsForOrigin.find(name);
    if (channelsForOriginIterator == channelsForOrigin.end())
        return;

    auto& channelIdentifiersForName = channelsForOriginIterator->value;
    if (!channelIdentifiersForName.removeFirst(identifier))
        return;
    if (!channelIdentifiersForName.isEmpty())
        return;

    channelsForOrigin.remove(channelsForOriginIterator);
    if (auto clientOrigin = toClientOrigin(origin))
        networkProcessConnection().send(Messages::NetworkBroadcastChannelRegistry::UnregisterChannel { *clientOrigin, name }, 0);

    if (channelsForOrigin.isEmpty())
        m_channelsPerOrigin.remove(channelsPerOriginIterator);
}

void WebBroadcastChannelRegistry::postMessage(const WebCore::PartitionedSecurityOrigin& origin, const String& name, WebCore::BroadcastChannelIdentifier source, Ref<WebCore::SerializedScriptValue>&& message, CompletionHandler<void()>&& completionHandler)
{
    auto callbackAggregator = CallbackAggregator::create(WTFMove(completionHandler));
    postMessageLocally(origin, name, source, message.copyRef(), callbackAggregator.copyRef());
    if (auto clientOrigin = toClientOrigin(origin))
        networkProcessConnection().sendWithAsyncReply(Messages::NetworkBroadcastChannelRegistry::PostMessage { *clientOrigin, name, WebCore::MessageWithMessagePorts { WTFMove(message), { } } }, [callbackAggregator] { }, 0);
}

void WebBroadcastChannelRegistry::postMessageLocally(const WebCore::PartitionedSecurityOrigin& origin, const String& name, std::optional<WebCore::BroadcastChannelIdentifier> sourceInProcess, Ref<WebCore::SerializedScriptValue>&& message, Ref<WTF::CallbackAggregator>&& callbackAggregator)
{
    auto channelsPerOriginIterator = m_channelsPerOrigin.find(origin);
    if (channelsPerOriginIterator == m_channelsPerOrigin.end())
        return;

    auto& channelsForOrigin = channelsPerOriginIterator->value;
    auto channelsForOriginIterator = channelsForOrigin.find(name);
    if (channelsForOriginIterator == channelsForOrigin.end())
        return;

    auto channelIdentifiersForName = channelsForOriginIterator->value;
    for (auto& channelIdentifier : channelIdentifiersForName) {
        if (channelIdentifier == sourceInProcess)
            continue;
        WebCore::BroadcastChannel::dispatchMessageTo(channelIdentifier, message.copyRef(), [callbackAggregator] { });
    }
}

void WebBroadcastChannelRegistry::postMessageToRemote(const WebCore::ClientOrigin& clientOrigin, const String& name, WebCore::MessageWithMessagePorts&& message, CompletionHandler<void()>&& completionHandler)
{
    auto callbackAggregator = CallbackAggregator::create(WTFMove(completionHandler));
    WebCore::PartitionedSecurityOrigin origin { clientOrigin.topOrigin.securityOrigin(), clientOrigin.clientOrigin.securityOrigin() };
    postMessageLocally(origin, name, std::nullopt, *message.message, callbackAggregator.copyRef());
}

void WebBroadcastChannelRegistry::networkProcessCrashed()
{
    for (auto& [origin, channelsForOrigin] : m_channelsPerOrigin) {
        auto clientOrigin = toClientOrigin(origin);
        if (!clientOrigin)
            continue;
        for (auto& name : channelsForOrigin.keys())
            networkProcessConnection().send(Messages::NetworkBroadcastChannelRegistry::RegisterChannel { *clientOrigin, name }, 0);
    }
}

} // namespace WebKit
