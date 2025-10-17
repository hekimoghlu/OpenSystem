/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#include "WebSpeechRecognitionConnection.h"

#include "MessageSenderInlines.h"
#include "SpeechRecognitionServerMessages.h"
#include "WebFrame.h"
#include "WebProcess.h"
#include "WebProcessProxyMessages.h"
#include "WebSpeechRecognitionConnectionMessages.h"
#include <WebCore/SpeechRecognitionConnectionClient.h>
#include <WebCore/SpeechRecognitionRequestInfo.h>
#include <WebCore/SpeechRecognitionUpdate.h>

namespace WebKit {

Ref<WebSpeechRecognitionConnection> WebSpeechRecognitionConnection::create(SpeechRecognitionConnectionIdentifier identifier)
{
    return adoptRef(*new WebSpeechRecognitionConnection(identifier));
}

WebSpeechRecognitionConnection::WebSpeechRecognitionConnection(SpeechRecognitionConnectionIdentifier identifier)
    : m_identifier(identifier)
{
    WebProcess::singleton().addMessageReceiver(Messages::WebSpeechRecognitionConnection::messageReceiverName(), m_identifier, *this);
    send(Messages::WebProcessProxy::CreateSpeechRecognitionServer(m_identifier), 0);

#if ENABLE(MEDIA_STREAM)
    WebProcess::singleton().ensureSpeechRecognitionRealtimeMediaSourceManager();
#endif
}

WebSpeechRecognitionConnection::~WebSpeechRecognitionConnection()
{
    send(Messages::WebProcessProxy::DestroySpeechRecognitionServer(m_identifier), 0);
    WebProcess::singleton().removeMessageReceiver(*this);
}

void WebSpeechRecognitionConnection::registerClient(WebCore::SpeechRecognitionConnectionClient& client)
{
    m_clientMap.add(client.identifier(), client);
}

void WebSpeechRecognitionConnection::unregisterClient(WebCore::SpeechRecognitionConnectionClient& client)
{
    m_clientMap.remove(client.identifier());
}

void WebSpeechRecognitionConnection::start(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier, const String& lang, bool continuous, bool interimResults, uint64_t maxAlternatives, WebCore::ClientOrigin&& clientOrigin, WebCore::FrameIdentifier frameIdentifier)
{
    send(Messages::SpeechRecognitionServer::Start(clientIdentifier, lang, continuous, interimResults, maxAlternatives, WTFMove(clientOrigin), frameIdentifier));
}

void WebSpeechRecognitionConnection::stop(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier)
{
    send(Messages::SpeechRecognitionServer::Stop(clientIdentifier));
}

void WebSpeechRecognitionConnection::abort(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier)
{
    send(Messages::SpeechRecognitionServer::Abort(clientIdentifier));
}

void WebSpeechRecognitionConnection::invalidate(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier)
{
    send(Messages::SpeechRecognitionServer::Invalidate(clientIdentifier));
}

void WebSpeechRecognitionConnection::didReceiveUpdate(WebCore::SpeechRecognitionUpdate&& update)
{
    auto clientIdentifier = update.clientIdentifier();
    if (!m_clientMap.contains(clientIdentifier))
        return;

    auto client = m_clientMap.get(clientIdentifier);
    if (!client) {
        m_clientMap.remove(clientIdentifier);
        // Inform server that client does not exist any more.
        invalidate(clientIdentifier);
        return;
    }

    switch (update.type()) {
    case WebCore::SpeechRecognitionUpdateType::Start:
        client->didStart();
        break;
    case WebCore::SpeechRecognitionUpdateType::AudioStart:
        client->didStartCapturingAudio();
        break;
    case WebCore::SpeechRecognitionUpdateType::SoundStart:
        client->didStartCapturingSound();
        break;
    case WebCore::SpeechRecognitionUpdateType::SpeechStart:
        client->didStartCapturingSpeech();
        break;
    case WebCore::SpeechRecognitionUpdateType::SpeechEnd:
        client->didStopCapturingSpeech();
        break;
    case WebCore::SpeechRecognitionUpdateType::SoundEnd:
        client->didStopCapturingSound();
        break;
    case WebCore::SpeechRecognitionUpdateType::AudioEnd:
        client->didStopCapturingAudio();
        break;
    case WebCore::SpeechRecognitionUpdateType::NoMatch:
        client->didFindNoMatch();
        break;
    case WebCore::SpeechRecognitionUpdateType::Result:
        client->didReceiveResult(update.result());
        break;
    case WebCore::SpeechRecognitionUpdateType::Error:
        client->didError(update.error());
        break;
    case WebCore::SpeechRecognitionUpdateType::End:
        client->didEnd();
    }
}

IPC::Connection* WebSpeechRecognitionConnection::messageSenderConnection() const
{
    return WebProcess::singleton().parentProcessConnection();
}

uint64_t WebSpeechRecognitionConnection::messageSenderDestinationID() const
{
    return m_identifier.toUInt64();
}

} // namespace WebKit
