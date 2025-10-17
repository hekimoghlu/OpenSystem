/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#include "SpeechSynthesisEvent.h"

#if ENABLE(SPEECH_SYNTHESIS)

#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SpeechSynthesisEvent);

Ref<SpeechSynthesisEvent> SpeechSynthesisEvent::create(const AtomString& type, const SpeechSynthesisEventInit& initializer)
{
    return adoptRef(*new SpeechSynthesisEvent(EventInterfaceType::SpeechSynthesisEvent, type, initializer));
}

SpeechSynthesisEvent::SpeechSynthesisEvent(enum EventInterfaceType eventInterface, const AtomString& type, const SpeechSynthesisEventInit& initializer)
    : Event(eventInterface, type, CanBubble::No, IsCancelable::No)
    , m_utterance(initializer.utterance)
    , m_charIndex(initializer.charIndex)
    , m_charLength(initializer.charLength)
    , m_elapsedTime(initializer.elapsedTime)
    , m_name(initializer.name)
{
}
    
} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
