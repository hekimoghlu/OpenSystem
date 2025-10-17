/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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

class RTCPeerConnectionIceErrorEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCPeerConnectionIceErrorEvent);
public:
    virtual ~RTCPeerConnectionIceErrorEvent();

    struct Init : EventInit {
        String address;
        std::optional<uint16_t> port;
        String url;
        uint16_t errorCode { 0 };
        String errorText;
    };

    static Ref<RTCPeerConnectionIceErrorEvent> create(const AtomString& type, Init&&);
    static Ref<RTCPeerConnectionIceErrorEvent> create(CanBubble, IsCancelable, String&& address, std::optional<uint16_t> port, String&& url, uint16_t errorCode, String&& errorText);

    const String& address() const { return m_address; }
    std::optional<uint16_t> port() const { return m_port; }
    const String& url() const { return m_url; }
    uint16_t errorCode() const { return m_errorCode; }
    const String& errorText() const { return m_errorText; }

private:
    RTCPeerConnectionIceErrorEvent(const AtomString& type, CanBubble, IsCancelable, String&& address, std::optional<uint16_t> port, String&& url, uint16_t errorCode, String&& errorText);

    String m_address;
    std::optional<uint16_t> m_port;
    String m_url;
    uint16_t m_errorCode { 0 };
    String m_errorText;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
