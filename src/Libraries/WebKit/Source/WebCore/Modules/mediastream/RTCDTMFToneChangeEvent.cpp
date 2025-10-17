/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
#include "RTCDTMFToneChangeEvent.h"

#if ENABLE(WEB_RTC)

#include "EventNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCDTMFToneChangeEvent);

Ref<RTCDTMFToneChangeEvent> RTCDTMFToneChangeEvent::create(const String& tone)
{
    return adoptRef(*new RTCDTMFToneChangeEvent(tone));
}

Ref<RTCDTMFToneChangeEvent> RTCDTMFToneChangeEvent::create(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new RTCDTMFToneChangeEvent(type, initializer, isTrusted));
}

RTCDTMFToneChangeEvent::RTCDTMFToneChangeEvent(const String& tone)
    : Event(EventInterfaceType::RTCDTMFToneChangeEvent, eventNames().tonechangeEvent, CanBubble::No, IsCancelable::No)
    , m_tone(tone)
{
}

RTCDTMFToneChangeEvent::RTCDTMFToneChangeEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::RTCDTMFToneChangeEvent, type, initializer, isTrusted)
    , m_tone(initializer.tone)
{
}

RTCDTMFToneChangeEvent::~RTCDTMFToneChangeEvent() = default;

const String& RTCDTMFToneChangeEvent::tone() const
{
    return m_tone;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)

