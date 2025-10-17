/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include <wtf/text/AtomString.h>

namespace WebCore {
class RTCIceCandidate;

class RTCPeerConnectionIceEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCPeerConnectionIceEvent);
public:
    virtual ~RTCPeerConnectionIceEvent();

    struct Init : EventInit {
        RefPtr<RTCIceCandidate> candidate;
        String url;
    };

    static Ref<RTCPeerConnectionIceEvent> create(const AtomString& type, Init&&);
    static Ref<RTCPeerConnectionIceEvent> create(CanBubble, IsCancelable, RefPtr<RTCIceCandidate>&&, String&& serverURL);

    RTCIceCandidate* candidate() const;
    const String& url() const { return m_url; }

private:
    RTCPeerConnectionIceEvent(const AtomString& type, CanBubble, IsCancelable, RefPtr<RTCIceCandidate>&&, String&& serverURL);

    RefPtr<RTCIceCandidate> m_candidate;
    String m_url;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
