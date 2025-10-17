/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "SpeechRecognitionServer.h"

#include "MessageSenderInlines.h"
#include "UserMediaProcessManager.h"
#include "WebProcessProxy.h"
#include "WebSpeechRecognitionConnectionMessages.h"
#include <WebCore/SpeechRecognitionRequestInfo.h>
#include <WebCore/SpeechRecognitionUpdate.h>
#include <wtf/TZoneMallocInlines.h>

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_BASE(assertion, *messageSenderConnection())

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SpeechRecognitionServer);

Ref<SpeechRecognitionServer> SpeechRecognitionServer::create(WebProcessProxy& process, SpeechRecognitionServerIdentifier identifier, SpeechRecognitionPermissionChecker&& permissionChecker, SpeechRecognitionCheckIfMockSpeechRecognitionEnabled&& checkIfEnabled
#if ENABLE(MEDIA_STREAM)
    , RealtimeMediaSourceCreateFunction&& function
#endif
    )
{
    return adoptRef(*new SpeechRecognitionServer(process, identifier, WTFMove(permissionChecker), WTFMove(checkIfEnabled)
#if ENABLE(MEDIA_STREAM)
        , WTFMove(function)
#endif
    ));
}

SpeechRecognitionServer::SpeechRecognitionServer(WebProcessProxy& process, SpeechRecognitionServerIdentifier identifier, SpeechRecognitionPermissionChecker&& permissionChecker, SpeechRecognitionCheckIfMockSpeechRecognitionEnabled&& checkIfEnabled
#if ENABLE(MEDIA_STREAM)
    , RealtimeMediaSourceCreateFunction&& function
#endif
    )
    : m_process(process)
    , m_identifier(identifier)
    , m_permissionChecker(WTFMove(permissionChecker))
    , m_checkIfMockSpeechRecognitionEnabled(WTFMove(checkIfEnabled))
#if ENABLE(MEDIA_STREAM)
    , m_realtimeMediaSourceCreateFunction(WTFMove(function))
#endif
{
}

std::optional<SharedPreferencesForWebProcess> SpeechRecognitionServer::sharedPreferencesForWebProcess() const
{
    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG return m_process ? m_process->sharedPreferencesForWebProcess() : std::nullopt;
}

void SpeechRecognitionServer::start(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier, String&& lang, bool continuous, bool interimResults, uint64_t maxAlternatives, WebCore::ClientOrigin&& origin, WebCore::FrameIdentifier frameIdentifier)
{
    ASSERT(!m_requests.contains(clientIdentifier));
    auto requestInfo = WebCore::SpeechRecognitionRequestInfo { clientIdentifier, WTFMove(lang), continuous, interimResults, maxAlternatives, WTFMove(origin), frameIdentifier };
    auto& newRequest = m_requests.add(clientIdentifier, makeUnique<WebCore::SpeechRecognitionRequest>(WTFMove(requestInfo))).iterator->value;

    requestPermissionForRequest(*newRequest);
}

void SpeechRecognitionServer::requestPermissionForRequest(WebCore::SpeechRecognitionRequest& request)
{
    m_permissionChecker(request, [weakThis = WeakPtr { *this }, weakRequest = WeakPtr { request }](auto error) mutable {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis || !weakRequest)
            return;

        auto identifier = weakRequest->clientIdentifier();
        auto request = protectedThis->m_requests.take(identifier);
        if (error) {
            protectedThis->sendUpdate(identifier, WebCore::SpeechRecognitionUpdateType::Error, WTFMove(error));
            return;
        }

        ASSERT(request);
        protectedThis->handleRequest(makeUniqueRefFromNonNullUniquePtr(WTFMove(request)));
    });
}

void SpeechRecognitionServer::handleRequest(UniqueRef<WebCore::SpeechRecognitionRequest>&& request)
{
    if (m_recognizer) {
        m_recognizer->abort(WebCore::SpeechRecognitionError { WebCore::SpeechRecognitionErrorType::Aborted, "Another request is started"_s });
        m_recognizer->prepareForDestruction();
    }

    auto clientIdentifier = request->clientIdentifier();
    m_recognizer = makeUnique<WebCore::SpeechRecognizer>([weakThis = WeakPtr { *this }](auto& update) {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;

        protectedThis->sendUpdate(update);

        if (update.type() == WebCore::SpeechRecognitionUpdateType::Error)
            protectedThis->m_recognizer->abort();
        else if (update.type() == WebCore::SpeechRecognitionUpdateType::End)
            protectedThis->m_recognizer->setInactive();
    }, WTFMove(request));

#if ENABLE(MEDIA_STREAM)
    auto sourceOrError = m_realtimeMediaSourceCreateFunction();
    if (!sourceOrError) {
        sendUpdate(WebCore::SpeechRecognitionUpdate::createError(clientIdentifier, WebCore::SpeechRecognitionError { WebCore::SpeechRecognitionErrorType::AudioCapture, sourceOrError.error.errorMessage }));
        return;
    }

    WebProcessProxy::muteCaptureInPagesExcept(m_identifier);
    bool mockDeviceCapturesEnabled = m_checkIfMockSpeechRecognitionEnabled();
    m_recognizer->start(sourceOrError.source(), mockDeviceCapturesEnabled);
#else
    sendUpdate(clientIdentifier, WebCore::SpeechRecognitionUpdateType::Error, WebCore::SpeechRecognitionError { WebCore::SpeechRecognitionErrorType::AudioCapture, "Audio capture is not implemented"_s });
#endif
}

void SpeechRecognitionServer::stop(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier)
{
    if (m_requests.remove(clientIdentifier)) {
        sendUpdate(clientIdentifier, WebCore::SpeechRecognitionUpdateType::End);
        return;
    }

    if (m_recognizer && m_recognizer->clientIdentifier() == clientIdentifier)
        m_recognizer->stop();
}

void SpeechRecognitionServer::abort(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier)
{
    if (m_requests.remove(clientIdentifier)) {
        sendUpdate(clientIdentifier, WebCore::SpeechRecognitionUpdateType::End);
        return;
    }

    if (m_recognizer && m_recognizer->clientIdentifier() == clientIdentifier)
        m_recognizer->abort();
}

void SpeechRecognitionServer::invalidate(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier)
{
    if (m_recognizer && m_recognizer->clientIdentifier() == clientIdentifier)
        m_recognizer->abort();
}

void SpeechRecognitionServer::sendUpdate(WebCore::SpeechRecognitionConnectionClientIdentifier clientIdentifier, WebCore::SpeechRecognitionUpdateType type, std::optional<WebCore::SpeechRecognitionError> error, std::optional<Vector<WebCore::SpeechRecognitionResultData>> result)
{
    auto update = WebCore::SpeechRecognitionUpdate::create(clientIdentifier, type);
    if (type == WebCore::SpeechRecognitionUpdateType::Error)
        update = WebCore::SpeechRecognitionUpdate::createError(clientIdentifier, *error);
    if (type == WebCore::SpeechRecognitionUpdateType::Result)
        update = WebCore::SpeechRecognitionUpdate::createResult(clientIdentifier, *result);
    sendUpdate(update);
}

void SpeechRecognitionServer::sendUpdate(const WebCore::SpeechRecognitionUpdate& update)
{
    send(Messages::WebSpeechRecognitionConnection::DidReceiveUpdate(update));
}

IPC::Connection* SpeechRecognitionServer::messageSenderConnection() const
{
    return m_process ? &m_process->connection() : nullptr;
}

uint64_t SpeechRecognitionServer::messageSenderDestinationID() const
{
    return m_identifier.toUInt64();
}

void SpeechRecognitionServer::mute()
{
    if (!m_recognizer)
        return;
    
    if (auto* source = m_recognizer->source())
        source->mute();
}

} // namespace WebKit

#undef MESSAGE_CHECK
