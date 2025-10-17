/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
#include "RTCDataChannelRemoteManager.h"

#if ENABLE(WEB_RTC)

#include "Connection.h"
#include "NetworkConnectionToWebProcessMessages.h"
#include "NetworkProcessConnection.h"
#include "RTCDataChannelRemoteManagerMessages.h"
#include "RTCDataChannelRemoteManagerProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/RTCDataChannel.h>
#include <WebCore/RTCError.h>
#include <WebCore/ScriptExecutionContext.h>

namespace WebKit {

RTCDataChannelRemoteManager& RTCDataChannelRemoteManager::singleton()
{
    static NeverDestroyed<Ref<RTCDataChannelRemoteManager>> sharedManager = [] {
        Ref instance = adoptRef(*new RTCDataChannelRemoteManager);
        instance->initialize();
        return instance;
    }();
    return sharedManager.get();
}

RTCDataChannelRemoteManager::RTCDataChannelRemoteManager()
    : m_queue(WorkQueue::create("RTCDataChannelRemoteManager"_s))
    , m_connection(&WebProcess::singleton().ensureNetworkProcessConnection().connection())
{
}

void RTCDataChannelRemoteManager::initialize()
{
    // FIXME: If the network process crashes, all RTC data will be misdelivered for the web process.
    // https://bugs.webkit.org/show_bug.cgi?id=245062
    m_connection->addMessageReceiver(m_queue, *this, Messages::RTCDataChannelRemoteManager::messageReceiverName());
}

bool RTCDataChannelRemoteManager::connectToRemoteSource(WebCore::RTCDataChannelIdentifier localIdentifier, WebCore::RTCDataChannelIdentifier remoteIdentifier)
{
    ASSERT(WebCore::Process::identifier() == localIdentifier.processIdentifier());
    if (WebCore::Process::identifier() != localIdentifier.processIdentifier())
        return false;

    auto handler = WebCore::RTCDataChannel::handlerFromIdentifier(localIdentifier.object());
    if (!handler)
        return false;

    auto iterator = m_sources.add(remoteIdentifier.object(), makeUniqueRef<WebCore::RTCDataChannelRemoteSource>(remoteIdentifier, makeUniqueRefFromNonNullUniquePtr(WTFMove(handler)), remoteSourceConnection()));
    return iterator.isNewEntry;
}

WebCore::RTCDataChannelRemoteHandlerConnection& RTCDataChannelRemoteManager::remoteHandlerConnection()
{
    if (!m_remoteHandlerConnection)
        m_remoteHandlerConnection = RemoteHandlerConnection::create(m_queue.copyRef());
    return *m_remoteHandlerConnection;
}

WebCore::RTCDataChannelRemoteSourceConnection& RTCDataChannelRemoteManager::remoteSourceConnection()
{
    if (!m_remoteSourceConnection)
        m_remoteSourceConnection = RemoteSourceConnection::create();
    return *m_remoteSourceConnection;
}

void RTCDataChannelRemoteManager::postTaskToHandler(WebCore::RTCDataChannelIdentifier handlerIdentifier, Function<void(WebCore::RTCDataChannelRemoteHandler&)>&& function)
{
    ASSERT(WebCore::Process::identifier() == handlerIdentifier.processIdentifier());
    if (WebCore::Process::identifier() != handlerIdentifier.processIdentifier())
        return;

    auto iterator = m_handlers.find(handlerIdentifier.object());
    if (iterator == m_handlers.end())
        return;
    auto& remoteHandler = iterator->value;

    WebCore::ScriptExecutionContext::postTaskTo(*remoteHandler.contextIdentifier, [handler = remoteHandler.handler, function = WTFMove(function)](auto&) mutable {
        if (handler)
            function(*handler);
    });
}

WebCore::RTCDataChannelRemoteSource* RTCDataChannelRemoteManager::sourceFromIdentifier(WebCore::RTCDataChannelIdentifier sourceIdentifier)
{
    ASSERT(WebCore::Process::identifier() == sourceIdentifier.processIdentifier());
    if (WebCore::Process::identifier() != sourceIdentifier.processIdentifier())
        return nullptr;

    return m_sources.get(sourceIdentifier.object());
}

void RTCDataChannelRemoteManager::sendData(WebCore::RTCDataChannelIdentifier sourceIdentifier, bool isRaw, std::span<const uint8_t> data)
{
    if (auto* source = sourceFromIdentifier(sourceIdentifier)) {
        if (isRaw)
            source->sendRawData(data);
        else
            source->sendStringData(CString(data));
    }
}

void RTCDataChannelRemoteManager::close(WebCore::RTCDataChannelIdentifier sourceIdentifier)
{
    if (auto* source = sourceFromIdentifier(sourceIdentifier))
        source->close();
}

void RTCDataChannelRemoteManager::changeReadyState(WebCore::RTCDataChannelIdentifier handlerIdentifier, WebCore::RTCDataChannelState state)
{
    postTaskToHandler(handlerIdentifier, [state](auto& handler) {
        handler.didChangeReadyState(state);
    });
}

void RTCDataChannelRemoteManager::receiveData(WebCore::RTCDataChannelIdentifier handlerIdentifier, bool isRaw, std::span<const uint8_t> data)
{
    Vector<uint8_t> buffer;
    String text;
    if (isRaw)
        buffer = Vector(data);
    else
        text = String::fromUTF8(data);

    postTaskToHandler(handlerIdentifier, [isRaw, text = WTFMove(text).isolatedCopy(), buffer = WTFMove(buffer)](auto& handler) mutable {
        if (isRaw)
            handler.didReceiveRawData(buffer.span());
        else
            handler.didReceiveStringData(WTFMove(text));
    });
}

void RTCDataChannelRemoteManager::detectError(WebCore::RTCDataChannelIdentifier handlerIdentifier, WebCore::RTCErrorDetailType detail, String&& message)
{
    postTaskToHandler(handlerIdentifier, [detail, message = WTFMove(message)](auto& handler) mutable {
        handler.didDetectError(WebCore::RTCError::create(detail, WTFMove(message)));
    });
}

void RTCDataChannelRemoteManager::bufferedAmountIsDecreasing(WebCore::RTCDataChannelIdentifier handlerIdentifier, size_t amount)
{
    postTaskToHandler(handlerIdentifier, [amount](auto& handler) {
        handler.bufferedAmountIsDecreasing(amount);
    });
}

Ref<RTCDataChannelRemoteManager::RemoteHandlerConnection> RTCDataChannelRemoteManager::RemoteHandlerConnection::create(Ref<WorkQueue>&& queue)
{
    return adoptRef(*new RemoteHandlerConnection(WTFMove(queue)));
}

RTCDataChannelRemoteManager::RemoteHandlerConnection::RemoteHandlerConnection(Ref<WorkQueue>&& queue)
    : m_connection(WebProcess::singleton().ensureNetworkProcessConnection().connection())
    , m_queue(WTFMove(queue))
{
}

void RTCDataChannelRemoteManager::RemoteHandlerConnection::connectToSource(WebCore::RTCDataChannelRemoteHandler& handler, std::optional<WebCore::ScriptExecutionContextIdentifier> contextIdentifier, WebCore::RTCDataChannelIdentifier localIdentifier, WebCore::RTCDataChannelIdentifier remoteIdentifier)
{
    m_queue->dispatch([handler = WeakPtr { handler }, contextIdentifier, localIdentifier]() mutable {
        RTCDataChannelRemoteManager::singleton().m_handlers.add(localIdentifier.object(), RemoteHandler { WTFMove(handler), *contextIdentifier });
    });
    m_connection->sendWithAsyncReply(Messages::NetworkConnectionToWebProcess::ConnectToRTCDataChannelRemoteSource { localIdentifier, remoteIdentifier }, [localIdentifier](auto&& result) {
        RTCDataChannelRemoteManager::singleton().postTaskToHandler(localIdentifier, [result](auto& handler) {
            if (!result || !*result) {
                handler.didDetectError(WebCore::RTCError::create(WebCore::RTCErrorDetailType::DataChannelFailure, "Unable to find data channel"_s));
                return;
            }
            handler.readyToSend();
        });
    }, 0);
}

void RTCDataChannelRemoteManager::RemoteHandlerConnection::sendData(WebCore::RTCDataChannelIdentifier identifier, bool isRaw, std::span<const uint8_t> data)
{
    m_connection->send(Messages::RTCDataChannelRemoteManagerProxy::SendData { identifier, isRaw, data }, 0);
}

void RTCDataChannelRemoteManager::RemoteHandlerConnection::close(WebCore::RTCDataChannelIdentifier identifier)
{
    // FIXME: We need to wait to send this message until RTCDataChannelRemoteManagerProxy::ConnectToSource is actually sent.
    m_connection->send(Messages::RTCDataChannelRemoteManagerProxy::Close { identifier }, 0);
}

Ref<RTCDataChannelRemoteManager::RemoteSourceConnection> RTCDataChannelRemoteManager::RemoteSourceConnection::create()
{
    return adoptRef(*new RemoteSourceConnection);
}

RTCDataChannelRemoteManager::RemoteSourceConnection::RemoteSourceConnection()
    : m_connection(WebProcess::singleton().ensureNetworkProcessConnection().connection())
{
}

void RTCDataChannelRemoteManager::RemoteSourceConnection::didChangeReadyState(WebCore::RTCDataChannelIdentifier identifier, WebCore::RTCDataChannelState state)
{
    m_connection->send(Messages::RTCDataChannelRemoteManagerProxy::ChangeReadyState { identifier, state }, 0);
}

void RTCDataChannelRemoteManager::RemoteSourceConnection::didReceiveStringData(WebCore::RTCDataChannelIdentifier identifier, const String& string)
{
    auto text = string.utf8();
    m_connection->send(Messages::RTCDataChannelRemoteManagerProxy::ReceiveData { identifier, false, text.span() }, 0);
}

void RTCDataChannelRemoteManager::RemoteSourceConnection::didReceiveRawData(WebCore::RTCDataChannelIdentifier identifier, std::span<const uint8_t> data)
{
    m_connection->send(Messages::RTCDataChannelRemoteManagerProxy::ReceiveData { identifier, true, data }, 0);
}

void RTCDataChannelRemoteManager::RemoteSourceConnection::didDetectError(WebCore::RTCDataChannelIdentifier identifier, WebCore::RTCErrorDetailType type, const String& message)
{
    m_connection->send(Messages::RTCDataChannelRemoteManagerProxy::DetectError { identifier, type, message }, 0);
}

void RTCDataChannelRemoteManager::RemoteSourceConnection::bufferedAmountIsDecreasing(WebCore::RTCDataChannelIdentifier identifier, size_t amount)
{
    m_connection->send(Messages::RTCDataChannelRemoteManagerProxy::BufferedAmountIsDecreasing { identifier, amount }, 0);
}

}

#endif
