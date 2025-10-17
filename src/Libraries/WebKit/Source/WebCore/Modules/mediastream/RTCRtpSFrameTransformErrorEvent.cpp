/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#include "RTCRtpSFrameTransformErrorEvent.h"

#if ENABLE(WEB_RTC)

#include "EventNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCRtpSFrameTransformErrorEvent);

Ref<RTCRtpSFrameTransformErrorEvent> RTCRtpSFrameTransformErrorEvent::create(CanBubble canBubble, IsCancelable isCancelable, Type errorType)
{
    return adoptRef(*new RTCRtpSFrameTransformErrorEvent(eventNames().errorEvent, canBubble, isCancelable, errorType));
}

Ref<RTCRtpSFrameTransformErrorEvent> RTCRtpSFrameTransformErrorEvent::create(const AtomString& type, Init&& init)
{
    return adoptRef(*new RTCRtpSFrameTransformErrorEvent(type, init.bubbles ? CanBubble::Yes : CanBubble::No,
        init.cancelable ? IsCancelable::Yes : IsCancelable::No, init.errorType));
}

RTCRtpSFrameTransformErrorEvent::RTCRtpSFrameTransformErrorEvent(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, Type errorType)
    : Event(EventInterfaceType::RTCRtpSFrameTransformErrorEvent, type, canBubble, cancelable)
    , m_errorType(errorType)
{
}

RTCRtpSFrameTransformErrorEvent::~RTCRtpSFrameTransformErrorEvent() = default;

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
