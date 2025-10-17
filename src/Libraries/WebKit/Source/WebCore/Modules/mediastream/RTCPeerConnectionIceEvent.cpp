/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#include "RTCPeerConnectionIceEvent.h"

#if ENABLE(WEB_RTC)

#include "EventNames.h"
#include "RTCIceCandidate.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCPeerConnectionIceEvent);

Ref<RTCPeerConnectionIceEvent> RTCPeerConnectionIceEvent::create(CanBubble canBubble, IsCancelable cancelable, RefPtr<RTCIceCandidate>&& candidate, String&& serverURL)
{
    return adoptRef(*new RTCPeerConnectionIceEvent(eventNames().icecandidateEvent, canBubble, cancelable, WTFMove(candidate), WTFMove(serverURL)));
}

Ref<RTCPeerConnectionIceEvent> RTCPeerConnectionIceEvent::create(const AtomString& type, Init&& init)
{
    return adoptRef(*new RTCPeerConnectionIceEvent(type, init.bubbles ? CanBubble::Yes : CanBubble::No,
        init.cancelable ? IsCancelable::Yes : IsCancelable::No, WTFMove(init.candidate), WTFMove(init.url)));
}

RTCPeerConnectionIceEvent::RTCPeerConnectionIceEvent(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, RefPtr<RTCIceCandidate>&& candidate, String&& serverURL)
    : Event(EventInterfaceType::RTCPeerConnectionIceEvent, type, canBubble, cancelable)
    , m_candidate(WTFMove(candidate))
    , m_url(WTFMove(serverURL))
{
}

RTCPeerConnectionIceEvent::~RTCPeerConnectionIceEvent() = default;

RTCIceCandidate* RTCPeerConnectionIceEvent::candidate() const
{
    return m_candidate.get();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
