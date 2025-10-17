/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#include "RTCPeerConnectionIceErrorEvent.h"

#if ENABLE(WEB_RTC)

#include "EventNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCPeerConnectionIceErrorEvent);

Ref<RTCPeerConnectionIceErrorEvent> RTCPeerConnectionIceErrorEvent::create(CanBubble canBubble, IsCancelable isCancelable, String&& address, std::optional<uint16_t> port, String&& url, uint16_t errorCode, String&& errorText)
{
    return adoptRef(*new RTCPeerConnectionIceErrorEvent(eventNames().icecandidateerrorEvent, canBubble, isCancelable, WTFMove(address), port, WTFMove(url), errorCode, WTFMove(errorText)));
}

Ref<RTCPeerConnectionIceErrorEvent> RTCPeerConnectionIceErrorEvent::create(const AtomString& type, Init&& init)
{
    return adoptRef(*new RTCPeerConnectionIceErrorEvent(type, init.bubbles ? CanBubble::Yes : CanBubble::No,
        init.cancelable ? IsCancelable::Yes : IsCancelable::No, WTFMove(init.address), init.port, WTFMove(init.url), WTFMove(init.errorCode), WTFMove(init.errorText)));
}

RTCPeerConnectionIceErrorEvent::RTCPeerConnectionIceErrorEvent(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, String&& address, std::optional<uint16_t> port, String&& url, uint16_t errorCode, String&& errorText)
    : Event(EventInterfaceType::RTCPeerConnectionIceErrorEvent, type, canBubble, cancelable)
    , m_address(WTFMove(address))
    , m_port(port)
    , m_url(WTFMove(url))
    , m_errorCode(errorCode)
    , m_errorText(WTFMove(errorText))
{
}

RTCPeerConnectionIceErrorEvent::~RTCPeerConnectionIceErrorEvent() = default;

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
