/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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

#if ENABLE(WEB_RTC)

#include "Event.h"
#include "RTCDataChannel.h"
#include <wtf/text/AtomString.h>

namespace WebCore {

class RTCDataChannelEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCDataChannelEvent);
public:
    struct Init : EventInit {
        RefPtr<RTCDataChannel> channel;
    };

    static Ref<RTCDataChannelEvent> create(const AtomString& type, CanBubble, IsCancelable, Ref<RTCDataChannel>&&);
    static Ref<RTCDataChannelEvent> create(const AtomString& type, Init&&, IsTrusted = IsTrusted::No);

    RTCDataChannel& channel();

private:
    RTCDataChannelEvent(const AtomString& type, CanBubble, IsCancelable, Ref<RTCDataChannel>&&);
    RTCDataChannelEvent(const AtomString& type, Init&&, IsTrusted);

    Ref<RTCDataChannel> m_channel;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
