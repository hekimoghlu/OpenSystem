/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
#include "RTCTrackEvent.h"

#if ENABLE(WEB_RTC)

#include "MediaStream.h"
#include "MediaStreamTrack.h"
#include "RTCRtpTransceiver.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCTrackEvent);

Ref<RTCTrackEvent> RTCTrackEvent::create(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, RefPtr<RTCRtpReceiver>&& receiver, RefPtr<MediaStreamTrack>&& track, Vector<Ref<MediaStream>>&& streams, RefPtr<RTCRtpTransceiver>&& transceiver)
{
    return adoptRef(*new RTCTrackEvent(type, canBubble, cancelable, WTFMove(receiver), WTFMove(track), WTFMove(streams), WTFMove(transceiver)));
}

Ref<RTCTrackEvent> RTCTrackEvent::create(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new RTCTrackEvent(type, initializer, isTrusted));
}

RTCTrackEvent::RTCTrackEvent(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, RefPtr<RTCRtpReceiver>&& receiver, RefPtr<MediaStreamTrack>&& track, Vector<Ref<MediaStream>>&& streams, RefPtr<RTCRtpTransceiver>&& transceiver)
    : Event(EventInterfaceType::RTCTrackEvent, type, canBubble, cancelable)
    , m_receiver(WTFMove(receiver))
    , m_track(WTFMove(track))
    , m_streams(WTFMove(streams))
    , m_transceiver(WTFMove(transceiver))
{
}

RTCTrackEvent::RTCTrackEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::RTCTrackEvent, type, initializer, isTrusted)
    , m_receiver(initializer.receiver)
    , m_track(initializer.track)
    , m_streams(initializer.streams)
    , m_transceiver(initializer.transceiver)
{
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
