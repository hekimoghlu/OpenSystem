/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
#include "SpeechSynthesisUtterance.h"

#if ENABLE(SPEECH_SYNTHESIS)

#include "ContextDestructionObserverInlines.h"
#include "EventNames.h"
#include "SpeechSynthesisErrorEvent.h"
#include "SpeechSynthesisEvent.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SpeechSynthesisUtterance);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SpeechSynthesisUtteranceActivity);

void SpeechSynthesisUtterance::ref() const
{
    RefCounted::ref();
}

void SpeechSynthesisUtterance::deref() const
{
    RefCounted::deref();
}

Ref<SpeechSynthesisUtterance> SpeechSynthesisUtterance::create(ScriptExecutionContext& context, const String& text)
{
    auto utterance = adoptRef(*new SpeechSynthesisUtterance(context, text, { }));
    utterance->suspendIfNeeded();
    return utterance;
}

Ref<SpeechSynthesisUtterance> SpeechSynthesisUtterance::create(ScriptExecutionContext& context, const String& text, SpeechSynthesisUtterance::UtteranceCompletionHandler&& completion)
{
    auto utterance = adoptRef(*new SpeechSynthesisUtterance(context, text, WTFMove(completion)));
    utterance->suspendIfNeeded();
    return utterance;
}

SpeechSynthesisUtterance::SpeechSynthesisUtterance(ScriptExecutionContext& context, const String& text, UtteranceCompletionHandler&& completion)
    : ActiveDOMObject(&context)
    , m_platformUtterance(PlatformSpeechSynthesisUtterance::create(*this))
    , m_completionHandler(WTFMove(completion))
{
    m_platformUtterance->setText(text);
}

SpeechSynthesisUtterance::~SpeechSynthesisUtterance()
{
    m_platformUtterance->setClient(nullptr);
}

SpeechSynthesisVoice* SpeechSynthesisUtterance::voice() const
{
    return m_voice.get();
}

void SpeechSynthesisUtterance::setVoice(SpeechSynthesisVoice* voice)
{
    if (!voice)
        return;
    
    // Cache our own version of the SpeechSynthesisVoice so that we don't have to do some lookup
    // to go from the platform voice back to the speech synthesis voice in the read property.
    m_voice = voice;
    
    if (voice)
        m_platformUtterance->setVoice(voice->platformVoice());
}

void SpeechSynthesisUtterance::eventOccurred(const AtomString& type, unsigned long charIndex, unsigned long charLength, const String& name)
{
    if (m_completionHandler) {
        if (type == eventNames().endEvent)
            m_completionHandler(*this);

        return;
    }

    dispatchEvent(SpeechSynthesisEvent::create(type, { this, charIndex, charLength, static_cast<float>((MonotonicTime::now() - startTime()).seconds()), name }));
}

void SpeechSynthesisUtterance::errorEventOccurred(const AtomString& type, SpeechSynthesisErrorCode errorCode)
{
    if (m_completionHandler) {
        m_completionHandler(*this);
        return;
    }

    dispatchEvent(SpeechSynthesisErrorEvent::create(type, { { this, 0, 0, static_cast<float>((MonotonicTime::now() - startTime()).seconds()), { } }, errorCode }));
}

void SpeechSynthesisUtterance::incrementActivityCountForEventDispatch()
{
    ++m_activityCountForEventDispatch;
}

void SpeechSynthesisUtterance::decrementActivityCountForEventDispatch()
{
    --m_activityCountForEventDispatch;
}

bool SpeechSynthesisUtterance::virtualHasPendingActivity() const
{
    return m_activityCountForEventDispatch && hasEventListeners();
}


} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
