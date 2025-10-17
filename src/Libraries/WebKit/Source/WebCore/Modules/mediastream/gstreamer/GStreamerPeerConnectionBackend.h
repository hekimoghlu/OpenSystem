/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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

#if USE(GSTREAMER_WEBRTC)

#include "GStreamerRtpSenderBackend.h"
#include "PeerConnectionBackend.h"
#include "RealtimeMediaSource.h"

#include <gst/gst.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class GStreamerPeerConnectionBackend;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::GStreamerPeerConnectionBackend> : std::true_type { };
}

namespace WebCore {

class GStreamerMediaEndpoint;
class GStreamerRtpReceiverBackend;
class GStreamerRtpTransceiverBackend;
class RTCRtpReceiver;
class RTCRtpReceiverBackend;
class RTCSessionDescription;
class RTCStatsReport;
class RealtimeIncomingAudioSourceGStreamer;
class RealtimeIncomingVideoSourceGStreamer;
class RealtimeMediaSourceGStreamer;
class RealtimeOutgoingAudioSourceGStreamer;
class RealtimeOutgoingVideoSourceGStreamer;

struct GStreamerIceCandidate {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    unsigned sdpMLineIndex;
    String candidate;
};

class GStreamerPeerConnectionBackend final : public PeerConnectionBackend {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerPeerConnectionBackend);
public:
    explicit GStreamerPeerConnectionBackend(RTCPeerConnection&);
    ~GStreamerPeerConnectionBackend();

    GStreamerRtpSenderBackend& backendFromRTPSender(RTCRtpSender&);

    void dispatchSenderBitrateRequest(const GRefPtr<GstWebRTCDTLSTransport>&, uint32_t bitrate);

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
    void getStats(Ref<DeferredPromise>&&) final;
    void getStats(RTCRtpSender&, Ref<DeferredPromise>&&) final;
    void getStats(RTCRtpReceiver&, Ref<DeferredPromise>&&) final;

    void emulatePlatformEvent(const String&) final { }
    void applyRotationForOutgoingVideoSources() final;

    void gatherDecoderImplementationName(Function<void(String&&)>&&) final;

    bool isNegotiationNeeded(uint32_t) const final;

    std::optional<bool> canTrickleIceCandidates() const final;

    void startGatheringStatLogs(Function<void(String&&)>&&) final;
    void stopGatheringStatLogs() final;
    void provideStatLogs(String&&);
    friend class RtcEventLogOutput;

    friend class GStreamerMediaEndpoint;
    friend class GStreamerRtpSenderBackend;
    RTCPeerConnection& connection();

    void getStatsSucceeded(const DeferredPromise&, Ref<RTCStatsReport>&&);

    ExceptionOr<Ref<RTCRtpSender>> addTrack(MediaStreamTrack&, FixedVector<String>&&) final;
    void removeTrack(RTCRtpSender&) final;

    ExceptionOr<Ref<RTCRtpTransceiver>> addTransceiver(const String&, const RTCRtpTransceiverInit&, IgnoreNegotiationNeededFlag) final;
    ExceptionOr<Ref<RTCRtpTransceiver>> addTransceiver(Ref<MediaStreamTrack>&&, const RTCRtpTransceiverInit&) final;

    GStreamerRtpSenderBackend::Source createSourceForTrack(MediaStreamTrack&);

    RTCRtpTransceiver* existingTransceiver(WTF::Function<bool(GStreamerRtpTransceiverBackend&)>&&);
    RTCRtpTransceiver& newRemoteTransceiver(std::unique_ptr<GStreamerRtpTransceiverBackend>&&, RealtimeMediaSource::Type, String&&);

    void collectTransceivers() final;

    bool isLocalDescriptionSet() const final { return m_isLocalDescriptionSet; }

    template<typename T>
    ExceptionOr<Ref<RTCRtpTransceiver>> addTransceiverFromTrackOrKind(T&& trackOrKind, const RTCRtpTransceiverInit&, IgnoreNegotiationNeededFlag);

    Ref<RTCRtpReceiver> createReceiver(std::unique_ptr<GStreamerRtpReceiverBackend>&&, const String& trackKind, const String& trackId);

    void suspend() final;
    void resume() final;

    void setReconfiguring(bool isReconfiguring) { m_isReconfiguring = isReconfiguring; }
    bool isReconfiguring() const { return m_isReconfiguring; }

    void tearDown();

    Ref<GStreamerMediaEndpoint> m_endpoint;
    bool m_isLocalDescriptionSet { false };
    bool m_isRemoteDescriptionSet { false };

    bool m_isReconfiguring { false };

    Function<void(String&&)> m_rtcStatsLogCallback;
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
