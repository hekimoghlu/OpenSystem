/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#include "SpeechRecognitionErrorEvent.h"

#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SpeechRecognitionErrorEvent);

Ref<SpeechRecognitionErrorEvent> SpeechRecognitionErrorEvent::create(const AtomString& type, Init&& init, IsTrusted isTrusted)
{
    return adoptRef(*new SpeechRecognitionErrorEvent(type, WTFMove(init), isTrusted));
}

Ref<SpeechRecognitionErrorEvent> SpeechRecognitionErrorEvent::create(const AtomString& type, SpeechRecognitionErrorCode error, const String& message)
{
    return adoptRef(*new SpeechRecognitionErrorEvent(type, error, message));
}

SpeechRecognitionErrorEvent::SpeechRecognitionErrorEvent(const AtomString& type, Init&& init, IsTrusted isTrusted)
    : Event(EventInterfaceType::SpeechRecognitionErrorEvent, type, init, isTrusted)
    , m_error(init.error)
    , m_message(WTFMove(init.message))
{
}

SpeechRecognitionErrorEvent::SpeechRecognitionErrorEvent(const AtomString& type, SpeechRecognitionErrorCode error, const String& message)
    : Event(EventInterfaceType::SpeechRecognitionErrorEvent, type, Event::CanBubble::No, Event::IsCancelable::No)
    , m_error(error)
    , m_message(message)
{
}

}; // namespace WebCore
