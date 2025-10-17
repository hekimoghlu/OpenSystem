/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#include "SpeechRecognitionEvent.h"

#include "SpeechRecognitionResultList.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SpeechRecognitionEvent);

Ref<SpeechRecognitionEvent> SpeechRecognitionEvent::create(const AtomString& type, Init&& init, IsTrusted isTrusted)
{
    return adoptRef(*new SpeechRecognitionEvent(type, WTFMove(init), isTrusted));
}

Ref<SpeechRecognitionEvent> SpeechRecognitionEvent::create(const AtomString& type, uint64_t resultIndex, RefPtr<SpeechRecognitionResultList>&& results)
{
    return adoptRef(*new SpeechRecognitionEvent(type, resultIndex, WTFMove(results)));
}

SpeechRecognitionEvent::SpeechRecognitionEvent(const AtomString& type, Init&& init, IsTrusted isTrusted)
    : Event(EventInterfaceType::SpeechRecognitionEvent, type, init, isTrusted)
    , m_resultIndex(init.resultIndex)
    , m_results(WTFMove(init.results))
{
}

SpeechRecognitionEvent::SpeechRecognitionEvent(const AtomString& type, uint64_t resultIndex, RefPtr<SpeechRecognitionResultList>&& results)
    : Event(EventInterfaceType::SpeechRecognitionEvent, type, Event::CanBubble::No, Event::IsCancelable::No)
    , m_resultIndex(resultIndex)
    , m_results(WTFMove(results))
{
}

SpeechRecognitionEvent::~SpeechRecognitionEvent() = default;

} // namespace WebCore
