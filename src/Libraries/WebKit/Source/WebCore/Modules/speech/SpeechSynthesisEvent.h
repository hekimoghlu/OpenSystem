/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

#include "Event.h"
#include "SpeechSynthesisEventInit.h"
#include "SpeechSynthesisUtterance.h"

namespace WebCore {

class SpeechSynthesisEvent : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SpeechSynthesisEvent);
public:
    
    static Ref<SpeechSynthesisEvent> create(const AtomString& type, const SpeechSynthesisEventInit&);

    const SpeechSynthesisUtterance* utterance() const { return m_utterance.get(); }
    unsigned long charIndex() const { return m_charIndex; }
    unsigned long charLength() const { return m_charLength; }
    float elapsedTime() const { return m_elapsedTime; }
    const String& name() const { return m_name; }

protected:
    SpeechSynthesisEvent(enum EventInterfaceType, const AtomString& type, const SpeechSynthesisEventInit&);

private:
    RefPtr<SpeechSynthesisUtterance> m_utterance;
    unsigned long m_charIndex;
    unsigned long m_charLength;
    float m_elapsedTime;
    String m_name;
};

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
