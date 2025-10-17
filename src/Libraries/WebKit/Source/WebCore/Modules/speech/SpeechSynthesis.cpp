/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
#include "SpeechSynthesis.h"

#if ENABLE(SPEECH_SYNTHESIS)

#include "Document.h"
#include "EventNames.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PlatformSpeechSynthesisVoice.h"
#include "PlatformSpeechSynthesizer.h"
#include "ScriptTelemetryCategory.h"
#include "SpeechSynthesisErrorEvent.h"
#include "SpeechSynthesisEvent.h"
#include "SpeechSynthesisUtterance.h"
#include "UserGestureIndicator.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SpeechSynthesis);

Ref<SpeechSynthesis> SpeechSynthesis::create(ScriptExecutionContext& context)
{
    auto synthesis = adoptRef(*new SpeechSynthesis(context));
    synthesis->suspendIfNeeded();
    return synthesis;
}

SpeechSynthesis::SpeechSynthesis(ScriptExecutionContext& context)
    : ActiveDOMObject(&context)
    , m_currentSpeechUtterance(nullptr)
    , m_isPaused(false)
    , m_restrictions(NoRestrictions)
    , m_speechSynthesisClient(nullptr)
{
    if (RefPtr document = dynamicDowncast<Document>(context)) {
#if PLATFORM(IOS_FAMILY)
        if (document->requiresUserGestureForAudioPlayback())
            m_restrictions = RequireUserGestureForSpeechStartRestriction;
#endif
        m_speechSynthesisClient = document->frame()->page()->speechSynthesisClient();
    }

    if (RefPtr speechSynthesisClient = m_speechSynthesisClient.get()) {
        speechSynthesisClient->setObserver(*this);
        speechSynthesisClient->resetState();
    }
}

SpeechSynthesis::~SpeechSynthesis() = default;

void SpeechSynthesis::setPlatformSynthesizer(Ref<PlatformSpeechSynthesizer>&& synthesizer)
{
    m_platformSpeechSynthesizer = synthesizer.ptr();
    if (m_voiceList)
        m_voiceList = std::nullopt;
    m_utteranceQueue.clear();
    // Finish current utterance.
    speakingErrorOccurred();
    m_isPaused = false;
    m_speechSynthesisClient = nullptr;
}

void SpeechSynthesis::voicesDidChange()
{
    if (m_voiceList)
        m_voiceList = std::nullopt;

    dispatchEvent(Event::create(eventNames().voiceschangedEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

PlatformSpeechSynthesizer& SpeechSynthesis::ensurePlatformSpeechSynthesizer()
{
    if (!m_platformSpeechSynthesizer)
        m_platformSpeechSynthesizer = PlatformSpeechSynthesizer::create(*this);
    return *m_platformSpeechSynthesizer;
}

const Vector<Ref<SpeechSynthesisVoice>>& SpeechSynthesis::getVoices()
{
    if (RefPtr context = scriptExecutionContext()) {
        if (context->requiresScriptExecutionTelemetry(ScriptTelemetryCategory::Speech)) {
            static NeverDestroyed<Vector<Ref<SpeechSynthesisVoice>>> emptyVoicesList;
            return emptyVoicesList.get();
        }
    }

    if (m_voiceList)
        return *m_voiceList;

    // If the voiceList is empty, that's the cue to get the voices from the platform again.
    RefPtr speechSynthesisClient = m_speechSynthesisClient.get();
    auto& voiceList = speechSynthesisClient ? speechSynthesisClient->voiceList() : ensurePlatformSpeechSynthesizer().voiceList();
    m_voiceList = voiceList.map([](auto& voice) {
        return SpeechSynthesisVoice::create(*voice);
    });

    return *m_voiceList;
}

bool SpeechSynthesis::speaking() const
{
    // If we have a current speech utterance, then that means we're assumed to be in a speaking state.
    // This state is independent of whether the utterance happens to be paused.
    return !!m_currentSpeechUtterance;
}

bool SpeechSynthesis::pending() const
{
    // This is true if there are any utterances that have not started.
    // That means there will be more than one in the queue.
    return m_utteranceQueue.size() > 1;
}

bool SpeechSynthesis::paused() const
{
    return m_isPaused;
}

void SpeechSynthesis::startSpeakingImmediately(SpeechSynthesisUtterance& utterance)
{
    utterance.setStartTime(MonotonicTime::now());
    m_currentSpeechUtterance = makeUnique<SpeechSynthesisUtteranceActivity>(Ref { utterance });
    m_isPaused = false;

    if (RefPtr speechSynthesisClient = m_speechSynthesisClient.get())
        speechSynthesisClient->speak(utterance.platformUtterance());
    else
        ensurePlatformSpeechSynthesizer().speak(utterance.platformUtterance());
}

void SpeechSynthesis::speak(SpeechSynthesisUtterance& utterance)
{
    // Like Audio, we should require that the user interact to start a speech synthesis session.
#if PLATFORM(IOS_FAMILY)
    if (UserGestureIndicator::processingUserGesture())
        removeBehaviorRestriction(RequireUserGestureForSpeechStartRestriction);
    else if (userGestureRequiredForSpeechStart())
        return;
#endif

    m_utteranceQueue.append(utterance);
    // If the queue was empty, speak this immediately and add it to the queue.
    if (m_utteranceQueue.size() == 1)
        startSpeakingImmediately(m_utteranceQueue.first());
}

void SpeechSynthesis::cancel()
{
    // Remove all the items from the utterance queue.
    // Hold on to the current utterance so the platform synthesizer can have a chance to clean up.
    RefPtr current = protectedCurrentSpeechUtterance();
    m_utteranceQueue.clear();
    if (RefPtr speechSynthesisClient = m_speechSynthesisClient.get()) {
        speechSynthesisClient->cancel();
        // If we wait for cancel to callback speakingErrorOccurred, then m_currentSpeechUtterance will be null
        // and the event won't be processed. Instead we process the error immediately.
        speakingErrorOccurred();
        m_currentSpeechUtterance = nullptr;
    } else if (m_platformSpeechSynthesizer)
        m_platformSpeechSynthesizer->cancel();

    current = nullptr;
}

void SpeechSynthesis::pause()
{
    if (!m_isPaused) {
        if (RefPtr speechSynthesisClient = m_speechSynthesisClient.get())
            speechSynthesisClient->pause();
        else if (m_platformSpeechSynthesizer)
            m_platformSpeechSynthesizer->pause();
    }
}

void SpeechSynthesis::resumeSynthesis()
{
    if (m_currentSpeechUtterance) {
        if (RefPtr speechSynthesisClient = m_speechSynthesisClient.get())
            speechSynthesisClient->resume();
        else if (m_platformSpeechSynthesizer)
            m_platformSpeechSynthesizer->resume();
    }
}

void SpeechSynthesis::handleSpeakingCompleted(SpeechSynthesisUtterance& utterance, bool errorOccurred)
{
    ASSERT(m_currentSpeechUtterance);
    Ref<SpeechSynthesisUtterance> protect(utterance);

    m_currentSpeechUtterance = nullptr;

    if (errorOccurred)
        utterance.errorEventOccurred(eventNames().errorEvent, SpeechSynthesisErrorCode::Canceled);
    else
        utterance.eventOccurred(eventNames().endEvent, 0, 0, String());
    
    if (m_utteranceQueue.size()) {
        Ref<SpeechSynthesisUtterance> firstUtterance = m_utteranceQueue.takeFirst();
        ASSERT(&utterance == firstUtterance.ptr());

        // Start the next job if there is one pending.
        if (!m_utteranceQueue.isEmpty())
            startSpeakingImmediately(m_utteranceQueue.first());
    }
}

void SpeechSynthesis::boundaryEventOccurred(PlatformSpeechSynthesisUtterance& platformUtterance, SpeechBoundary boundary, unsigned charIndex, unsigned charLength)
{
    static NeverDestroyed<const String> wordBoundaryString(MAKE_STATIC_STRING_IMPL("word"));
    static NeverDestroyed<const String> sentenceBoundaryString(MAKE_STATIC_STRING_IMPL("sentence"));

    ASSERT(platformUtterance.client());

    auto utterance = static_cast<SpeechSynthesisUtterance*>(platformUtterance.client());
    switch (boundary) {
    case SpeechBoundary::SpeechWordBoundary:
        utterance->eventOccurred(eventNames().boundaryEvent, charIndex, charLength, wordBoundaryString);
        break;
    case SpeechBoundary::SpeechSentenceBoundary:
        utterance->eventOccurred(eventNames().boundaryEvent, charIndex, charLength, sentenceBoundaryString);
        break;
    default:
        ASSERT_NOT_REACHED();
    }
}

void SpeechSynthesis::didStartSpeaking()
{
    if (!m_currentSpeechUtterance)
        return;
    didStartSpeaking(*protectedCurrentSpeechUtterance()->platformUtterance());
}

void SpeechSynthesis::didFinishSpeaking()
{
    if (!m_currentSpeechUtterance)
        return;
    didFinishSpeaking(*protectedCurrentSpeechUtterance()->platformUtterance());
}

void SpeechSynthesis::didPauseSpeaking()
{
    if (!m_currentSpeechUtterance)
        return;
    didPauseSpeaking(*protectedCurrentSpeechUtterance()->platformUtterance());
}

void SpeechSynthesis::didResumeSpeaking()
{
    if (!m_currentSpeechUtterance)
        return;
    didResumeSpeaking(*protectedCurrentSpeechUtterance()->platformUtterance());
}

void SpeechSynthesis::speakingErrorOccurred()
{
    if (!m_currentSpeechUtterance)
        return;
    speakingErrorOccurred(*protectedCurrentSpeechUtterance()->platformUtterance());
}

void SpeechSynthesis::boundaryEventOccurred(bool wordBoundary, unsigned charIndex, unsigned charLength)
{
    if (!m_currentSpeechUtterance)
        return;
    boundaryEventOccurred(*protectedCurrentSpeechUtterance()->platformUtterance(), wordBoundary ? SpeechBoundary::SpeechWordBoundary : SpeechBoundary::SpeechSentenceBoundary, charIndex, charLength);
}

void SpeechSynthesis::voicesChanged()
{
    voicesDidChange();
}

void SpeechSynthesis::didStartSpeaking(PlatformSpeechSynthesisUtterance& utterance)
{
    if (utterance.client())
        static_cast<SpeechSynthesisUtterance&>(*utterance.client()).eventOccurred(eventNames().startEvent, 0, 0, String());
}

void SpeechSynthesis::didPauseSpeaking(PlatformSpeechSynthesisUtterance& utterance)
{
    m_isPaused = true;
    if (utterance.client())
        static_cast<SpeechSynthesisUtterance&>(*utterance.client()).eventOccurred(eventNames().pauseEvent, 0, 0, String());
}

void SpeechSynthesis::didResumeSpeaking(PlatformSpeechSynthesisUtterance& utterance)
{
    m_isPaused = false;
    if (utterance.client())
        static_cast<SpeechSynthesisUtterance&>(*utterance.client()).eventOccurred(eventNames().resumeEvent, 0, 0, String());
}

void SpeechSynthesis::didFinishSpeaking(PlatformSpeechSynthesisUtterance& utterance)
{
    if (utterance.client())
        handleSpeakingCompleted(static_cast<SpeechSynthesisUtterance&>(*utterance.client()), false);
}

void SpeechSynthesis::speakingErrorOccurred(PlatformSpeechSynthesisUtterance& utterance)
{
    if (utterance.client())
        handleSpeakingCompleted(static_cast<SpeechSynthesisUtterance&>(*utterance.client()), true);
}

RefPtr<SpeechSynthesisUtterance> SpeechSynthesis::protectedCurrentSpeechUtterance()
{
    return m_currentSpeechUtterance ? &m_currentSpeechUtterance->utterance() : nullptr;
}

void SpeechSynthesis::simulateVoicesListChange()
{
    if (m_speechSynthesisClient) {
        voicesChanged();
        return;
    }

    if (m_platformSpeechSynthesizer)
        voicesDidChange();
}

bool SpeechSynthesis::virtualHasPendingActivity() const
{
    return m_voiceList && m_hasEventListener;
}

void SpeechSynthesis::eventListenersDidChange()
{
    m_hasEventListener = hasEventListeners(eventNames().voiceschangedEvent);
}

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
