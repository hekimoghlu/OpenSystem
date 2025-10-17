/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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

#if ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)

#include "GRefPtrGStreamer.h"
#include "RTCRtpReceiverBackend.h"
#include "RealtimeMediaSource.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GStreamerRtpReceiverBackend final : public RTCRtpReceiverBackend {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerRtpReceiverBackend);
public:
    explicit GStreamerRtpReceiverBackend(GRefPtr<GstWebRTCRTPTransceiver>&&);

    GstWebRTCRTPReceiver* rtcReceiver() { return m_rtcReceiver.get(); }
    Ref<RealtimeMediaSource> createSource(const String& trackKind, const String& trackId);

private:
    RTCRtpParameters getParameters() final;
    Vector<RTCRtpContributingSource> getContributingSources() const final;
    Vector<RTCRtpSynchronizationSource> getSynchronizationSources() const final;
    Ref<RTCRtpTransformBackend> rtcRtpTransformBackend() final;
    std::unique_ptr<RTCDtlsTransportBackend> dtlsTransportBackend() final;

    GRefPtr<GstWebRTCRTPReceiver> m_rtcReceiver;
    GRefPtr<GstWebRTCRTPTransceiver> m_rtcTransceiver;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
