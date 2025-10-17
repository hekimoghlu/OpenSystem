/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
#include "RTCDTMFSender.h"

#if ENABLE(WEB_RTC)

#include "ContextDestructionObserverInlines.h"
#include "RTCDTMFSenderBackend.h"
#include "RTCDTMFToneChangeEvent.h"
#include "RTCRtpSender.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCDTMFSender);

static const size_t minToneDurationMs = 40;
static const size_t maxToneDurationMs = 6000;
static const size_t minInterToneGapMs = 30;

Ref<RTCDTMFSender> RTCDTMFSender::create(ScriptExecutionContext& context, RTCRtpSender& sender, std::unique_ptr<RTCDTMFSenderBackend>&& backend)
{
    auto result = adoptRef(*new RTCDTMFSender(context, sender, WTFMove(backend)));
    result->suspendIfNeeded();
    return result;
}

RTCDTMFSender::RTCDTMFSender(ScriptExecutionContext& context, RTCRtpSender& sender, std::unique_ptr<RTCDTMFSenderBackend>&& backend)
    : ActiveDOMObject(&context)
    , m_toneTimer(*this, &RTCDTMFSender::toneTimerFired)
    , m_sender(sender)
    , m_backend(WTFMove(backend))
{
    m_backend->onTonePlayed([this] {
        onTonePlayed();
    });
}

RTCDTMFSender::~RTCDTMFSender() = default;

bool RTCDTMFSender::canInsertDTMF() const
{
    if (!m_sender || m_sender->isStopped())
        return false;

    auto currentDirection = m_sender->currentTransceiverDirection();
    if (!currentDirection)
        return false;
    if (*currentDirection != RTCRtpTransceiverDirection::Sendrecv && *currentDirection != RTCRtpTransceiverDirection::Sendonly)
        return false;

    return m_backend && m_backend->canInsertDTMF();
}

String RTCDTMFSender::toneBuffer() const
{
    return m_tones;
}

static inline bool isToneCharacterInvalid(UChar character)
{
    if (character >= '0' && character <= '9')
        return false;
    if (character >= 'A' && character <= 'D')
        return false;
    return character != '#' && character != '*' && character != ',';
}

ExceptionOr<void> RTCDTMFSender::insertDTMF(const String& tones, size_t duration, size_t interToneGap)
{
    if (!canInsertDTMF())
        return Exception { ExceptionCode::InvalidStateError, "Cannot insert DTMF"_s };

    auto normalizedTones = tones.convertToUppercaseWithoutLocale();
    if (normalizedTones.find(isToneCharacterInvalid) != notFound)
        return Exception { ExceptionCode::InvalidCharacterError, "Tones are not valid"_s };

    m_tones = WTFMove(normalizedTones);
    m_duration = clampTo(duration, minToneDurationMs, maxToneDurationMs);
    m_interToneGap = std::max(interToneGap, minInterToneGapMs);

    if (m_tones.isEmpty() || m_isPendingPlayoutTask)
        return { };

    m_isPendingPlayoutTask = true;
    scriptExecutionContext()->postTask([protectedThis = Ref { *this }](auto&) {
        protectedThis->playNextTone();
    });
    return { };
}

void RTCDTMFSender::playNextTone()
{
    if (m_tones.isEmpty()) {
        m_isPendingPlayoutTask = false;
        dispatchEvent(RTCDTMFToneChangeEvent::create({ }));
        return;
    }

    if (!canInsertDTMF()) {
        m_isPendingPlayoutTask = false;
        return;
    }

    auto currentTone = m_tones.left(1);
    m_tones = m_tones.substring(1);

    m_backend->playTone(currentTone[0], m_duration, m_interToneGap);
    dispatchEvent(RTCDTMFToneChangeEvent::create(currentTone));
}

void RTCDTMFSender::onTonePlayed()
{
    m_toneTimer.startOneShot(1_ms * m_interToneGap);
}

void RTCDTMFSender::toneTimerFired()
{
    playNextTone();
}

void RTCDTMFSender::stop()
{
    m_isPendingPlayoutTask = false;
    m_backend = nullptr;
    m_toneTimer.stop();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
