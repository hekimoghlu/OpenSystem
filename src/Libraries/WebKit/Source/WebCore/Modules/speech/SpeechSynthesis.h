/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#pragma once

#if ENABLE(SPEECH_SYNTHESIS)

#include "PlatformSpeechSynthesisUtterance.h"
#include "PlatformSpeechSynthesizer.h"
#include "SpeechSynthesisClient.h"
#include "SpeechSynthesisErrorCode.h"
#include "SpeechSynthesisUtterance.h"
#include "SpeechSynthesisVoice.h"
#include <wtf/Deque.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class Document;
class PlatformSpeechSynthesizerClient;
class SpeechSynthesisVoice;

class SpeechSynthesis : public PlatformSpeechSynthesizerClient, public SpeechSynthesisClientObserver, public RefCounted<SpeechSynthesis>, public ActiveDOMObject, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SpeechSynthesis);
public:
    static Ref<SpeechSynthesis> create(ScriptExecutionContext&);
    virtual ~SpeechSynthesis();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    bool pending() const;
    bool speaking() const;
    bool paused() const;

    void speak(SpeechSynthesisUtterance&);
    void cancel();
    void pause();
    void resumeSynthesis();

    const Vector<Ref<SpeechSynthesisVoice>>& getVoices();

    // Used in testing to use a mock platform synthesizer
    WEBCORE_EXPORT void setPlatformSynthesizer(Ref<PlatformSpeechSynthesizer>&&);

    // Restrictions to change default behaviors.
    enum BehaviorRestrictionFlags {
        NoRestrictions = 0,
        RequireUserGestureForSpeechStartRestriction = 1 << 0,
    };
    typedef unsigned BehaviorRestrictions;

    bool userGestureRequiredForSpeechStart() const { return m_restrictions & RequireUserGestureForSpeechStartRestriction; }
    void removeBehaviorRestriction(BehaviorRestrictions restriction) { m_restrictions &= ~restriction; }
    WEBCORE_EXPORT void simulateVoicesListChange();

private:
    SpeechSynthesis(ScriptExecutionContext&);
    RefPtr<SpeechSynthesisUtterance> protectedCurrentSpeechUtterance();

    // PlatformSpeechSynthesizerClient
    void voicesDidChange() override;
    void didStartSpeaking(PlatformSpeechSynthesisUtterance&) override;
    void didPauseSpeaking(PlatformSpeechSynthesisUtterance&) override;
    void didResumeSpeaking(PlatformSpeechSynthesisUtterance&) override;
    void didFinishSpeaking(PlatformSpeechSynthesisUtterance&) override;
    void speakingErrorOccurred(PlatformSpeechSynthesisUtterance&) override;
    void boundaryEventOccurred(PlatformSpeechSynthesisUtterance&, SpeechBoundary, unsigned charIndex, unsigned charLength) override;

    // SpeechSynthesisClientObserver
    void didStartSpeaking() override;
    void didFinishSpeaking() override;
    void didPauseSpeaking() override;
    void didResumeSpeaking() override;
    void speakingErrorOccurred() override;
    void boundaryEventOccurred(bool wordBoundary, unsigned charIndex, unsigned charLength) override;
    void voicesChanged() override;

    // ActiveDOMObject
    bool virtualHasPendingActivity() const final;

    void startSpeakingImmediately(SpeechSynthesisUtterance&);
    void handleSpeakingCompleted(SpeechSynthesisUtterance&, bool errorOccurred);

    // EventTarget
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::SpeechSynthesis; }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    void eventListenersDidChange() final;
    
    PlatformSpeechSynthesizer& ensurePlatformSpeechSynthesizer();
    
    RefPtr<PlatformSpeechSynthesizer> m_platformSpeechSynthesizer;
    std::optional<Vector<Ref<SpeechSynthesisVoice>>> m_voiceList;
    std::unique_ptr<SpeechSynthesisUtteranceActivity> m_currentSpeechUtterance;
    Deque<Ref<SpeechSynthesisUtterance>> m_utteranceQueue;
    bool m_isPaused;
    BehaviorRestrictions m_restrictions;
    WeakPtr<SpeechSynthesisClient> m_speechSynthesisClient;
    bool m_hasEventListener { false };
};

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
