/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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

class MediaStream;
class MediaStreamTrack;
class RTCRtpReceiver;
class RTCRtpTransceiver;

typedef Vector<Ref<MediaStream>> MediaStreamArray;

class RTCTrackEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCTrackEvent);
public:
    static Ref<RTCTrackEvent> create(const AtomString& type, CanBubble, IsCancelable, RefPtr<RTCRtpReceiver>&&, RefPtr<MediaStreamTrack>&&, MediaStreamArray&&, RefPtr<RTCRtpTransceiver>&&);

    struct Init : EventInit {
        RefPtr<RTCRtpReceiver> receiver;
        RefPtr<MediaStreamTrack> track;
        MediaStreamArray streams;
        RefPtr<RTCRtpTransceiver> transceiver;
    };
    static Ref<RTCTrackEvent> create(const AtomString& type, const Init&, IsTrusted = IsTrusted::No);

    RTCRtpReceiver* receiver() const { return m_receiver.get(); }
    MediaStreamTrack* track() const  { return m_track.get(); }
    const MediaStreamArray& streams() const  { return m_streams; }
    RTCRtpTransceiver* transceiver() const  { return m_transceiver.get(); }

private:
    RTCTrackEvent(const AtomString& type, CanBubble, IsCancelable, RefPtr<RTCRtpReceiver>&&, RefPtr<MediaStreamTrack>&&, MediaStreamArray&&, RefPtr<RTCRtpTransceiver>&&);
    RTCTrackEvent(const AtomString& type, const Init&, IsTrusted);

    RefPtr<RTCRtpReceiver> m_receiver;
    RefPtr<MediaStreamTrack> m_track;
    MediaStreamArray m_streams;
    RefPtr<RTCRtpTransceiver> m_transceiver;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
