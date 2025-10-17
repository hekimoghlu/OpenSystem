/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#include "RTCDataChannelEvent.h"

#if ENABLE(WEB_RTC)

#include "RTCDataChannel.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCDataChannelEvent);

Ref<RTCDataChannelEvent> RTCDataChannelEvent::create(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, Ref<RTCDataChannel>&& channel)
{
    return adoptRef(*new RTCDataChannelEvent(type, canBubble, cancelable, WTFMove(channel)));
}

Ref<RTCDataChannelEvent> RTCDataChannelEvent::create(const AtomString& type, Init&& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new RTCDataChannelEvent(type, WTFMove(initializer), isTrusted));
}

RTCDataChannelEvent::RTCDataChannelEvent(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, Ref<RTCDataChannel>&& channel)
    : Event(EventInterfaceType::RTCDataChannelEvent, type, canBubble, cancelable)
    , m_channel(WTFMove(channel))
{
}

RTCDataChannelEvent::RTCDataChannelEvent(const AtomString& type, Init&& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::RTCDataChannelEvent, type, initializer, isTrusted)
    , m_channel(initializer.channel.releaseNonNull())
{
}

RTCDataChannel& RTCDataChannelEvent::channel()
{
    return m_channel.get();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)

