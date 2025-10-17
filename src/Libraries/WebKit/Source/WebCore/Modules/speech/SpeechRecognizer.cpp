/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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
#include "SpeechRecognizer.h"

#include "SpeechRecognitionRequest.h"
#include "SpeechRecognitionUpdate.h"
#include <wtf/MediaTime.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include "MediaUtilities.h"
#include <pal/cf/CoreMediaSoftLink.h>
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SpeechRecognizer);

SpeechRecognizer::SpeechRecognizer(DelegateCallback&& delegateCallback, UniqueRef<SpeechRecognitionRequest>&& request)
    : m_delegateCallback(WTFMove(delegateCallback))
    , m_request(WTFMove(request))
#if HAVE(SPEECHRECOGNIZER)
    , m_currentAudioSampleTime(PAL::kCMTimeZero)
#endif
{
}

void SpeechRecognizer::abort(std::optional<SpeechRecognitionError>&& error)
{
    if (m_state == State::Aborting || m_state == State::Inactive)
        return;
    m_state = State::Aborting;

    if (error)
        m_delegateCallback(SpeechRecognitionUpdate::createError(clientIdentifier(), *error));

    stopCapture();
    abortRecognition();
}

void SpeechRecognizer::stop()
{
    if (m_state == State::Aborting || m_state == State::Inactive)
        return;
    m_state = State::Stopping;

    stopCapture();
    stopRecognition();
}

SpeechRecognitionConnectionClientIdentifier SpeechRecognizer::clientIdentifier() const
{
    return m_request->clientIdentifier();
}

void SpeechRecognizer::prepareForDestruction()
{
    if (m_state == State::Inactive)
        return;

    auto delegateCallback = std::exchange(m_delegateCallback, [](const SpeechRecognitionUpdate&) { });
    delegateCallback(SpeechRecognitionUpdate::create(clientIdentifier(), SpeechRecognitionUpdateType::End));
    m_state = State::Inactive;
}

#if ENABLE(MEDIA_STREAM)

void SpeechRecognizer::start(Ref<RealtimeMediaSource>&& source, bool mockSpeechRecognitionEnabled)
{
    if (!startRecognition(mockSpeechRecognitionEnabled, clientIdentifier(), m_request->lang(), m_request->continuous(), m_request->interimResults(), m_request->maxAlternatives())) {
        auto error = SpeechRecognitionError { SpeechRecognitionErrorType::ServiceNotAllowed, "Failed to start recognition"_s };
        m_delegateCallback(SpeechRecognitionUpdate::createError(clientIdentifier(), WTFMove(error)));
        return;
    }

    m_state = State::Running;
    m_delegateCallback(SpeechRecognitionUpdate::create(clientIdentifier(), SpeechRecognitionUpdateType::Start));
    startCapture(WTFMove(source));
}

void SpeechRecognizer::startCapture(Ref<RealtimeMediaSource>&& source)
{
    auto dataCallback = [weakThis = WeakPtr { *this }](const auto& time, const auto& data, const auto& description, auto sampleCount) {
        if (weakThis)
            weakThis->dataCaptured(time, data, description, sampleCount);
    };

    auto stateUpdateCallback = [weakThis = WeakPtr { *this }](const auto& update) {
        if (weakThis)
            weakThis->m_delegateCallback(update);
    };

    m_source = makeUnique<SpeechRecognitionCaptureSource>(clientIdentifier(), WTFMove(dataCallback), WTFMove(stateUpdateCallback), WTFMove(source));
}

#endif

void SpeechRecognizer::stopCapture()
{
    if (!m_source)
        return;

    m_source = nullptr;
    m_delegateCallback(SpeechRecognitionUpdate::create(clientIdentifier(), SpeechRecognitionUpdateType::AudioEnd));
}

#if !HAVE(SPEECHRECOGNIZER)

void SpeechRecognizer::dataCaptured(const MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t)
{
}

bool SpeechRecognizer::startRecognition(bool, SpeechRecognitionConnectionClientIdentifier, const String&, bool, bool, uint64_t)
{
    return true;
}

void SpeechRecognizer::abortRecognition()
{
    m_delegateCallback(SpeechRecognitionUpdate::create(clientIdentifier(), SpeechRecognitionUpdateType::End));
}

void SpeechRecognizer::stopRecognition()
{
    m_delegateCallback(SpeechRecognitionUpdate::create(clientIdentifier(), SpeechRecognitionUpdateType::End));
}

#endif

} // namespace WebCore
