/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#include "RTCDataChannelRemoteManagerProxy.h"

#if ENABLE(WEB_RTC)

#include "NetworkConnectionToWebProcess.h"
#include "RTCDataChannelRemoteManagerMessages.h"
#include "RTCDataChannelRemoteManagerProxyMessages.h"

namespace WebKit {

RTCDataChannelRemoteManagerProxy::RTCDataChannelRemoteManagerProxy()
    : m_queue(WorkQueue::create("RTCDataChannelRemoteManagerProxy"_s))
{
}

Ref<WorkQueue> RTCDataChannelRemoteManagerProxy::protectedQueue()
{
    return m_queue;
}

void RTCDataChannelRemoteManagerProxy::registerConnectionToWebProcess(NetworkConnectionToWebProcess& connectionToWebProcess)
{
    protectedQueue()->dispatch([this, protectedThis = Ref { *this }, identifier = connectionToWebProcess.webProcessIdentifier(), connectionID = connectionToWebProcess.connection().uniqueID()]() mutable {
        ASSERT(!m_webProcessConnections.contains(identifier));
        m_webProcessConnections.add(identifier, connectionID);
    });
    connectionToWebProcess.protectedConnection()->addWorkQueueMessageReceiver(Messages::RTCDataChannelRemoteManagerProxy::messageReceiverName(), m_queue, *this);
}

void RTCDataChannelRemoteManagerProxy::unregisterConnectionToWebProcess(NetworkConnectionToWebProcess& connectionToWebProcess)
{
    protectedQueue()->dispatch([this, protectedThis = Ref { *this }, identifier = connectionToWebProcess.webProcessIdentifier()] {
        ASSERT(m_webProcessConnections.contains(identifier));
        m_webProcessConnections.remove(identifier);
    });
    connectionToWebProcess.protectedConnection()->removeWorkQueueMessageReceiver(Messages::RTCDataChannelRemoteManagerProxy::messageReceiverName());
}

void RTCDataChannelRemoteManagerProxy::sendData(WebCore::RTCDataChannelIdentifier identifier, bool isRaw, std::span<const uint8_t> data)
{
    if (auto connectionID = m_webProcessConnections.getOptional(identifier.processIdentifier()))
        IPC::Connection::send(*connectionID, Messages::RTCDataChannelRemoteManager::SendData { identifier, isRaw, data }, 0);
}

void RTCDataChannelRemoteManagerProxy::close(WebCore::RTCDataChannelIdentifier identifier)
{
    if (auto connectionID = m_webProcessConnections.getOptional(identifier.processIdentifier()))
        IPC::Connection::send(*connectionID, Messages::RTCDataChannelRemoteManager::Close { identifier }, 0);
}

void RTCDataChannelRemoteManagerProxy::changeReadyState(WebCore::RTCDataChannelIdentifier identifier, WebCore::RTCDataChannelState state)
{
    if (auto connectionID = m_webProcessConnections.getOptional(identifier.processIdentifier()))
        IPC::Connection::send(*connectionID, Messages::RTCDataChannelRemoteManager::ChangeReadyState { identifier, state }, 0);
}

void RTCDataChannelRemoteManagerProxy::receiveData(WebCore::RTCDataChannelIdentifier identifier, bool isRaw, std::span<const uint8_t> data)
{
    if (auto connectionID = m_webProcessConnections.getOptional(identifier.processIdentifier()))
        IPC::Connection::send(*connectionID, Messages::RTCDataChannelRemoteManager::ReceiveData { identifier, isRaw, data }, 0);
}

void RTCDataChannelRemoteManagerProxy::detectError(WebCore::RTCDataChannelIdentifier identifier, WebCore::RTCErrorDetailType detail, const String& message)
{
    if (auto connectionID = m_webProcessConnections.getOptional(identifier.processIdentifier()))
        IPC::Connection::send(*connectionID, Messages::RTCDataChannelRemoteManager::DetectError { identifier, detail, message }, 0);
}

void RTCDataChannelRemoteManagerProxy::bufferedAmountIsDecreasing(WebCore::RTCDataChannelIdentifier identifier, size_t amount)
{
    if (auto connectionID = m_webProcessConnections.getOptional(identifier.processIdentifier()))
        IPC::Connection::send(*connectionID, Messages::RTCDataChannelRemoteManager::BufferedAmountIsDecreasing { identifier, amount }, 0);
}

}

#endif
