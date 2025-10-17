/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "LibWebRTCPeerConnectionBackend.h"
#include "RTCRtpSenderBackend.h"
#include "RealtimeOutgoingAudioSource.h"
#include "RealtimeOutgoingVideoSource.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

ALLOW_UNUSED_PARAMETERS_BEGIN

#include <webrtc/api/rtp_sender_interface.h>
#include <webrtc/api/scoped_refptr.h>

ALLOW_UNUSED_PARAMETERS_END

namespace WebCore {
class LibWebRTCRtpSenderBackend;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::LibWebRTCRtpSenderBackend> : std::true_type { };
}

namespace WebCore {

class LibWebRTCPeerConnectionBackend;

class LibWebRTCRtpSenderBackend final : public RTCRtpSenderBackend, public CanMakeWeakPtr<LibWebRTCRtpSenderBackend> {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCRtpSenderBackend);
public:
    using Source = std::variant<std::nullptr_t, Ref<RealtimeOutgoingAudioSource>, Ref<RealtimeOutgoingVideoSource>>;
    LibWebRTCRtpSenderBackend(LibWebRTCPeerConnectionBackend&, rtc::scoped_refptr<webrtc::RtpSenderInterface>&&, Source&&);
    LibWebRTCRtpSenderBackend(LibWebRTCPeerConnectionBackend&, rtc::scoped_refptr<webrtc::RtpSenderInterface>&&);
    ~LibWebRTCRtpSenderBackend();

    void setRTCSender(rtc::scoped_refptr<webrtc::RtpSenderInterface>&& rtcSender) { m_rtcSender = WTFMove(rtcSender); }
    webrtc::RtpSenderInterface* rtcSender() { return m_rtcSender.get(); }

    RealtimeOutgoingVideoSource* videoSource();
    void clearSource() { setSource(nullptr); }
    void setSource(Source&&);
    void takeSource(LibWebRTCRtpSenderBackend&);

private:
    bool replaceTrack(RTCRtpSender&, MediaStreamTrack*) final;
    RTCRtpSendParameters getParameters() const final;
    void setParameters(const RTCRtpSendParameters&, DOMPromiseDeferred<void>&&) final;
    std::unique_ptr<RTCDTMFSenderBackend> createDTMFBackend() final;
    Ref<RTCRtpTransformBackend> rtcRtpTransformBackend() final;
    std::unique_ptr<RTCDtlsTransportBackend> dtlsTransportBackend() final;
    void setMediaStreamIds(const FixedVector<String>&) final;

    void startSource();
    void stopSource();
    bool hasSource() const;

    RefPtr<LibWebRTCPeerConnectionBackend> protectedPeerConnectionBackend() const;

    WeakPtr<LibWebRTCPeerConnectionBackend> m_peerConnectionBackend;
    rtc::scoped_refptr<webrtc::RtpSenderInterface> m_rtcSender;
    Source m_source;
    RefPtr<RTCRtpTransformBackend> m_transformBackend;
    mutable std::optional<webrtc::RtpParameters> m_currentParameters;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
