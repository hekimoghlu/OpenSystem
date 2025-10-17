/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#include "MessagePortChannelProviderImpl.h"

#include "MessagePort.h"
#include <wtf/MainThread.h>
#include <wtf/RunLoop.h>

namespace WebCore {

MessagePortChannelProviderImpl::MessagePortChannelProviderImpl() = default;

MessagePortChannelProviderImpl::~MessagePortChannelProviderImpl()
{
    ASSERT_NOT_REACHED();
}

void MessagePortChannelProviderImpl::createNewMessagePortChannel(const MessagePortIdentifier& local, const MessagePortIdentifier& remote)
{
    ensureOnMainThread([weakRegistry = WeakPtr { m_registry }, local, remote] {
        if (CheckedPtr registry = weakRegistry.get())
            registry->didCreateMessagePortChannel(local, remote);
    });
}

void MessagePortChannelProviderImpl::entangleLocalPortInThisProcessToRemote(const MessagePortIdentifier& local, const MessagePortIdentifier& remote)
{
    ensureOnMainThread([weakRegistry = WeakPtr { m_registry }, local, remote] {
        if (CheckedPtr registry = weakRegistry.get())
            registry->didEntangleLocalToRemote(local, remote, Process::identifier());
    });
}

void MessagePortChannelProviderImpl::messagePortDisentangled(const MessagePortIdentifier& local)
{
    ensureOnMainThread([weakRegistry = WeakPtr { m_registry }, local] {
        if (CheckedPtr registry = weakRegistry.get())
            registry->didDisentangleMessagePort(local);
    });
}

void MessagePortChannelProviderImpl::messagePortClosed(const MessagePortIdentifier& local)
{
    ensureOnMainThread([weakRegistry = WeakPtr { m_registry }, local] {
        if (CheckedPtr registry = weakRegistry.get())
            registry->didCloseMessagePort(local);
    });
}

void MessagePortChannelProviderImpl::postMessageToRemote(MessageWithMessagePorts&& message, const MessagePortIdentifier& remoteTarget)
{
    ensureOnMainThread([weakRegistry = WeakPtr { m_registry }, message = WTFMove(message), remoteTarget]() mutable {
        CheckedPtr registry = weakRegistry.get();
        if (!registry)
            return;
        if (registry->didPostMessageToRemote(WTFMove(message), remoteTarget))
            MessagePort::notifyMessageAvailable(remoteTarget);
    });
}

void MessagePortChannelProviderImpl::takeAllMessagesForPort(const MessagePortIdentifier& port, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&& outerCallback)
{
    // It is the responsibility of outerCallback to get itself to the appropriate thread (e.g. WebWorker thread)
    auto callback = [outerCallback = WTFMove(outerCallback)](Vector<MessageWithMessagePorts>&& messages, CompletionHandler<void()>&& messageDeliveryCallback) mutable {
        ASSERT(isMainThread());
        outerCallback(WTFMove(messages), WTFMove(messageDeliveryCallback));
    };

    ensureOnMainThread([weakRegistry = WeakPtr { m_registry }, port, callback = WTFMove(callback)]() mutable {
        if (CheckedPtr registry = weakRegistry.get())
            registry->takeAllMessagesForPort(port, WTFMove(callback));
    });
}

} // namespace WebCore
