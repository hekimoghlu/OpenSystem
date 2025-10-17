/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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

#include "PeerConnectionBackend.h"
#include "RealtimeMediaSource.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class LibWebRTCPeerConnectionBackend;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::LibWebRTCPeerConnectionBackend> : std::true_type { };
}

namespace webrtc {
class IceCandidateInterface;
}

namespace WebCore {

class LibWebRTCMediaEndpoint;
class LibWebRTCProvider;
class LibWebRTCRtpSenderBackend;
class LibWebRTCRtpReceiverBackend;
class LibWebRTCRtpTransceiverBackend;
class RTCRtpReceiver;
class RTCRtpReceiverBackend;
class RTCSessionDescription;
class RTCStatsReport;
class RealtimeIncomingAudioSource;
class RealtimeIncomingVideoSource;
class RealtimeMediaSource;
class RealtimeOutgoingAudioSource;
class RealtimeOutgoingVideoSource;

class LibWebRTCPeerConnectionBackend final : public PeerConnectionBackend {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCPeerConnectionBackend);
public:
    LibWebRTCPeerConnectionBackend(RTCPeerConnection&, LibWebRTCProvider&);
    ~LibWebRTCPeerConnectionBackend();

    bool shouldEnableWebRTCL4S() const;

private:
    void close() final;
    void doCreateOffer(RTCOfferOptions&&) final;
    void doCreateAnswer(RTCAnswerOptions&&) final;
    void doSetLocalDescription(const RTCSessionDescription*) final;
    void doSetRemoteDescription(const RTCSessionDescription&) final;
    void doAddIceCandidate(RTCIceCandidate&, AddIceCandidateCallback&&) final;
    void doStop() final;
    std::unique_ptr<RTCDataChannelHandler> createDataChannelHandler(const String&, const RTCDataChannelInit&) final;
    void restartIce() final;
    bool setConfiguration(MediaEndpointConfiguration&&) final;
    void gatherDecoderImplementationName(Function<void(String&&)>&&) final;
    void getStats(Ref<DeferredPromise>&&) final;
    void getStats(RTCRtpSender&, Ref<DeferredPromise>&&) final;
    void getStats(RTCRtpReceiver&, Ref<DeferredPromise>&&) final;

    std::optional<bool> canTrickleIceCandidates() const final;

    void emulatePlatformEvent(const String&) final { }
    void applyRotationForOutgoingVideoSources() final;

    friend class LibWebRTCMediaEndpoint;
    friend class LibWebRTCRtpSenderBackend;
    RTCPeerConnection& connection() { return m_peerConnection; }

    void getStatsSucceeded(const DeferredPromise&, Ref<RTCStatsReport>&&);

    ExceptionOr<Ref<RTCRtpSender>> addTrack(MediaStreamTrack&, FixedVector<String>&&) final;
    void removeTrack(RTCRtpSender&) final;

    ExceptionOr<Ref<RTCRtpTransceiver>> addTransceiver(const String&, const RTCRtpTransceiverInit&, IgnoreNegotiationNeededFlag) final;
    ExceptionOr<Ref<RTCRtpTransceiver>> addTransceiver(Ref<MediaStreamTrack>&&, const RTCRtpTransceiverInit&) final;
    void setSenderSourceFromTrack(LibWebRTCRtpSenderBackend&, MediaStreamTrack&);

    RTCRtpTransceiver* existingTransceiver(Function<bool(LibWebRTCRtpTransceiverBackend&)>&&);
    RTCRtpTransceiver& newRemoteTransceiver(std::unique_ptr<LibWebRTCRtpTransceiverBackend>&&, RealtimeMediaSource::Type);

    void collectTransceivers() final;

private:
    bool isLocalDescriptionSet() const final { return m_isLocalDescriptionSet; }

    void startGatheringStatLogs(Function<void(String&&)>&&) final;
    void stopGatheringStatLogs() final;
    void provideStatLogs(String&&);
    friend class RtcEventLogOutput;

    template<typename T>
    ExceptionOr<Ref<RTCRtpTransceiver>> addTransceiverFromTrackOrKind(T&& trackOrKind, const RTCRtpTransceiverInit&, IgnoreNegotiationNeededFlag = IgnoreNegotiationNeededFlag::No);

    Ref<RTCRtpReceiver> createReceiver(std::unique_ptr<LibWebRTCRtpReceiverBackend>&&);

    void suspend() final;
    void resume() final;
    void disableICECandidateFiltering() final;
    bool isNegotiationNeeded(uint32_t) const final;

    Ref<LibWebRTCMediaEndpoint> m_endpoint;
    bool m_isLocalDescriptionSet { false };
    bool m_isRemoteDescriptionSet { false };

    Vector<std::unique_ptr<webrtc::IceCandidateInterface>> m_pendingCandidates;
    Vector<Ref<RTCRtpReceiver>> m_pendingReceivers;

    Function<void(String&&)> m_rtcStatsLogCallback;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
