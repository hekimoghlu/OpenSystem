/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "PlatformSpeechSynthesisUtterance.h"
#include "SpeechSynthesisErrorCode.h"
#include "SpeechSynthesisVoice.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WEBCORE_EXPORT SpeechSynthesisUtterance final : public PlatformSpeechSynthesisUtteranceClient, public RefCounted<SpeechSynthesisUtterance>, public ActiveDOMObject, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(SpeechSynthesisUtterance, WEBCORE_EXPORT);
public:
    void ref() const final;
    void deref() const final;

    using UtteranceCompletionHandler = Function<void(const SpeechSynthesisUtterance&)>;
    static Ref<SpeechSynthesisUtterance> create(ScriptExecutionContext&, const String&, UtteranceCompletionHandler&&);
    static Ref<SpeechSynthesisUtterance> create(ScriptExecutionContext&, const String&);

    // Create an empty default constructor so SpeechSynthesisEventInit compiles.
    SpeechSynthesisUtterance();

    virtual ~SpeechSynthesisUtterance();

    const String& text() const { return m_platformUtterance->text(); }
    void setText(const String& text) { m_platformUtterance->setText(text); }

    const String& lang() const { return m_platformUtterance->lang(); }
    void setLang(const String& lang) { m_platformUtterance->setLang(lang); }

    SpeechSynthesisVoice* voice() const;
    void setVoice(SpeechSynthesisVoice*);

    float volume() const { return m_platformUtterance->volume(); }
    void setVolume(float volume) { m_platformUtterance->setVolume(volume); }

    float rate() const { return m_platformUtterance->rate(); }
    void setRate(float rate) { m_platformUtterance->setRate(rate); }

    float pitch() const { return m_platformUtterance->pitch(); }
    void setPitch(float pitch) { m_platformUtterance->setPitch(pitch); }

    MonotonicTime startTime() const { return m_platformUtterance->startTime(); }
    void setStartTime(MonotonicTime startTime) { m_platformUtterance->setStartTime(startTime); }

    PlatformSpeechSynthesisUtterance* platformUtterance() const { return m_platformUtterance.get(); }

    void eventOccurred(const AtomString& type, unsigned long charIndex, unsigned long charLength, const String& name);
    void errorEventOccurred(const AtomString& type, SpeechSynthesisErrorCode);
    void setIsActiveForEventDispatch(bool);

private:
    SpeechSynthesisUtterance(ScriptExecutionContext&, const String&, UtteranceCompletionHandler&&);
    void dispatchEventAndUpdateState(Event&);
    void incrementActivityCountForEventDispatch();
    void decrementActivityCountForEventDispatch();

    // ActiveDOMObject
    bool virtualHasPendingActivity() const final;

    // EventTarget
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::SpeechSynthesisUtterance; }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    friend class SpeechSynthesisUtteranceActivity;
    RefPtr<PlatformSpeechSynthesisUtterance> m_platformUtterance;
    RefPtr<SpeechSynthesisVoice> m_voice;
    UtteranceCompletionHandler m_completionHandler;
    unsigned m_activityCountForEventDispatch { 0 };
};

class SpeechSynthesisUtteranceActivity {
    WTF_MAKE_TZONE_ALLOCATED(SpeechSynthesisUtteranceActivity);
public:
    SpeechSynthesisUtteranceActivity(Ref<SpeechSynthesisUtterance>&& utterance)
        : m_utterance(utterance)
    {
        m_utterance->incrementActivityCountForEventDispatch();
    }

    ~SpeechSynthesisUtteranceActivity()
    {
        m_utterance->decrementActivityCountForEventDispatch();
    }

    SpeechSynthesisUtterance& utterance() { return m_utterance.get(); }

private:
    Ref<SpeechSynthesisUtterance> m_utterance;
};

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
