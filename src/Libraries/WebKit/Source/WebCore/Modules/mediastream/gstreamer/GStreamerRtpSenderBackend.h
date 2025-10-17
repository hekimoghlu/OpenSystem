/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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

#include "GUniquePtrGStreamer.h"
#include "RTCRtpSenderBackend.h"
#include "RealtimeOutgoingAudioSourceGStreamer.h"
#include "RealtimeOutgoingVideoSourceGStreamer.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class GStreamerRtpSenderBackend;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::GStreamerRtpSenderBackend> : std::true_type { };
}

namespace WebCore {

class GStreamerPeerConnectionBackend;

class GStreamerRtpSenderBackend final : public RTCRtpSenderBackend {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerRtpSenderBackend);
public:
    GStreamerRtpSenderBackend(GStreamerPeerConnectionBackend&, GRefPtr<GstWebRTCRTPSender>&&);
    using Source = std::variant<std::nullptr_t, Ref<RealtimeOutgoingAudioSourceGStreamer>, Ref<RealtimeOutgoingVideoSourceGStreamer>>;
    GStreamerRtpSenderBackend(GStreamerPeerConnectionBackend&, GRefPtr<GstWebRTCRTPSender>&&, Source&&, GUniquePtr<GstStructure>&& initData);

    void setRTCSender(GRefPtr<GstWebRTCRTPSender>&& rtcSender) { m_rtcSender = WTFMove(rtcSender); }
    GstWebRTCRTPSender* rtcSender() { return m_rtcSender.get(); }

    RealtimeOutgoingAudioSourceGStreamer* audioSource()
    {
        return WTF::switchOn(m_source,
            [] (Ref<RealtimeOutgoingAudioSourceGStreamer>& source) { return source.ptr(); },
            [] (const auto&) -> RealtimeOutgoingAudioSourceGStreamer* { return nullptr; }
        );
    }

    RealtimeOutgoingVideoSourceGStreamer* videoSource()
    {
        return WTF::switchOn(m_source,
            [] (Ref<RealtimeOutgoingVideoSourceGStreamer>& source) { return source.ptr(); },
            [] (const auto&) -> RealtimeOutgoingVideoSourceGStreamer* { return nullptr; }
        );
    }

    bool hasSource() const
    {
        return WTF::switchOn(m_source,
            [] (const std::nullptr_t&) { return false; },
            [] (const auto&) { return true; }
        );
    }

    void clearSource();
    void setSource(Source&&);
    void takeSource(GStreamerRtpSenderBackend&);

    void stopSource();
    void tearDown();

    void dispatchBitrateRequest(uint32_t bitrate);

private:
    bool replaceTrack(RTCRtpSender&, MediaStreamTrack*) final;
    RTCRtpSendParameters getParameters() const final;
    void setParameters(const RTCRtpSendParameters&, DOMPromiseDeferred<void>&&) final;
    std::unique_ptr<RTCDTMFSenderBackend> createDTMFBackend() final;
    Ref<RTCRtpTransformBackend> rtcRtpTransformBackend() final;
    void setMediaStreamIds(const FixedVector<String>&) final;
    std::unique_ptr<RTCDtlsTransportBackend> dtlsTransportBackend() final;

    void startSource();

    WeakPtr<GStreamerPeerConnectionBackend> m_peerConnectionBackend;
    GRefPtr<GstWebRTCRTPSender> m_rtcSender;
    Source m_source;
    GUniquePtr<GstStructure> m_initData;
    mutable GUniquePtr<GstStructure> m_currentParameters;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
