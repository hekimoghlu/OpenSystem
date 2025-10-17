/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#include "WebMessagePortChannelProvider.h"

#include "NetworkConnectionToWebProcessMessages.h"
#include "NetworkProcessConnection.h"
#include "WebProcess.h"
#include <WebCore/MessagePort.h>
#include <WebCore/MessagePortIdentifier.h>
#include <WebCore/MessageWithMessagePorts.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebMessagePortChannelProvider);

WebMessagePortChannelProvider& WebMessagePortChannelProvider::singleton()
{
    static WebMessagePortChannelProvider* provider = new WebMessagePortChannelProvider;
    return *provider;
}

WebMessagePortChannelProvider::WebMessagePortChannelProvider()
{
}

WebMessagePortChannelProvider::~WebMessagePortChannelProvider()
{
    ASSERT_NOT_REACHED();
}

static inline IPC::Connection& networkProcessConnection()
{
    return WebProcess::singleton().ensureNetworkProcessConnection().connection();
}

void WebMessagePortChannelProvider::createNewMessagePortChannel(const MessagePortIdentifier& port1, const MessagePortIdentifier& port2)
{
    ASSERT(!m_inProcessPortMessages.contains(port1));
    ASSERT(!m_inProcessPortMessages.contains(port2));
    m_inProcessPortMessages.add(port1, Vector<MessageWithMessagePorts> { });
    m_inProcessPortMessages.add(port2, Vector<MessageWithMessagePorts> { });

    networkProcessConnection().send(Messages::NetworkConnectionToWebProcess::CreateNewMessagePortChannel { port1, port2 }, 0);
}

void WebMessagePortChannelProvider::entangleLocalPortInThisProcessToRemote(const MessagePortIdentifier& local, const MessagePortIdentifier& remote)
{
    m_inProcessPortMessages.add(local, Vector<MessageWithMessagePorts> { });

    networkProcessConnection().send(Messages::NetworkConnectionToWebProcess::EntangleLocalPortInThisProcessToRemote { local, remote }, 0);
}

void WebMessagePortChannelProvider::messagePortDisentangled(const MessagePortIdentifier& port)
{
    networkProcessConnection().send(Messages::NetworkConnectionToWebProcess::MessagePortDisentangled { port }, 0);
}

void WebMessagePortChannelProvider::messagePortSentToRemote(const WebCore::MessagePortIdentifier& port)
{
    auto inProcessPortMessages = m_inProcessPortMessages.take(port);
    for (auto& message : inProcessPortMessages)
        postMessageToRemote(WTFMove(message), port);
}

void WebMessagePortChannelProvider::messagePortClosed(const MessagePortIdentifier& port)
{
    m_inProcessPortMessages.remove(port);
    networkProcessConnection().send(Messages::NetworkConnectionToWebProcess::MessagePortClosed { port }, 0);
}

void WebMessagePortChannelProvider::takeAllMessagesForPort(const MessagePortIdentifier& port, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&& completionHandler)
{
    networkProcessConnection().sendWithAsyncReply(Messages::NetworkConnectionToWebProcess::TakeAllMessagesForPort { port }, [completionHandler = WTFMove(completionHandler), port](Vector<WebCore::MessageWithMessagePorts>&& messages, std::optional<MessageBatchIdentifier> messageBatchIdentifier) mutable {
        if (!messageBatchIdentifier)
            return completionHandler({ }, [] { }); // IPC failure.

        auto& inProcessPortMessages = WebMessagePortChannelProvider::singleton().m_inProcessPortMessages;
        auto iterator = inProcessPortMessages.find(port);
        if (iterator != inProcessPortMessages.end()) {
            auto pendingMessages = std::exchange(iterator->value, { });
            messages.appendVector(WTFMove(pendingMessages));
        }
        completionHandler(WTFMove(messages), [messageBatchIdentifier] {
            networkProcessConnection().send(Messages::NetworkConnectionToWebProcess::DidDeliverMessagePortMessages { *messageBatchIdentifier }, 0);
        });
    }, 0);
}

void WebMessagePortChannelProvider::postMessageToRemote(MessageWithMessagePorts&& message, const MessagePortIdentifier& remoteTarget)
{
    auto iterator = m_inProcessPortMessages.find(remoteTarget);
    if (iterator != m_inProcessPortMessages.end()) {
        iterator->value.append(WTFMove(message));
        WebProcess::singleton().messagesAvailableForPort(remoteTarget);
        return;
    }

    for (auto& port : message.transferredPorts)
        messagePortSentToRemote(port.first);

    networkProcessConnection().send(Messages::NetworkConnectionToWebProcess::PostMessageToRemote { message, remoteTarget }, 0);
}

} // namespace WebKit
