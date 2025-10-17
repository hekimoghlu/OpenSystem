/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#ifndef API_TEST_MOCK_PEERCONNECTIONINTERFACE_H_
#define API_TEST_MOCK_PEERCONNECTIONINTERFACE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "api/adaptation/resource.h"
#include "api/candidate.h"
#include "api/data_channel_interface.h"
#include "api/dtls_transport_interface.h"
#include "api/jsep.h"
#include "api/make_ref_counted.h"
#include "api/media_stream_interface.h"
#include "api/media_types.h"
#include "api/peer_connection_interface.h"
#include "api/rtc_error.h"
#include "api/rtc_event_log_output.h"
#include "api/rtp_parameters.h"
#include "api/rtp_receiver_interface.h"
#include "api/rtp_sender_interface.h"
#include "api/rtp_transceiver_interface.h"
#include "api/scoped_refptr.h"
#include "api/sctp_transport_interface.h"
#include "api/set_remote_description_observer_interface.h"
#include "api/stats/rtc_stats_collector_callback.h"
#include "api/transport/bandwidth_estimation_settings.h"
#include "api/transport/bitrate_settings.h"
#include "api/transport/network_control.h"
#include "rtc_base/ref_counted_object.h"
#include "test/gmock.h"

namespace webrtc {

class MockPeerConnectionInterface : public webrtc::PeerConnectionInterface {
 public:
  static rtc::scoped_refptr<MockPeerConnectionInterface> Create() {
    return rtc::make_ref_counted<MockPeerConnectionInterface>();
  }

  // PeerConnectionInterface
  MOCK_METHOD(rtc::scoped_refptr<StreamCollectionInterface>,
              local_streams,
              (),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<StreamCollectionInterface>,
              remote_streams,
              (),
              (override));
  MOCK_METHOD(bool, AddStream, (MediaStreamInterface*), (override));
  MOCK_METHOD(void, RemoveStream, (MediaStreamInterface*), (override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<RtpSenderInterface>>,
              AddTrack,
              (rtc::scoped_refptr<MediaStreamTrackInterface>,
               const std::vector<std::string>&),
              (override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<RtpSenderInterface>>,
              AddTrack,
              (rtc::scoped_refptr<MediaStreamTrackInterface>,
               const std::vector<std::string>&,
               const std::vector<RtpEncodingParameters>&),
              (override));
  MOCK_METHOD(RTCError,
              RemoveTrackOrError,
              (rtc::scoped_refptr<RtpSenderInterface>),
              (override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              (rtc::scoped_refptr<MediaStreamTrackInterface>),
              (override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              (rtc::scoped_refptr<MediaStreamTrackInterface>,
               const RtpTransceiverInit&),
              (override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              (cricket::MediaType),
              (override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              (cricket::MediaType, const RtpTransceiverInit&),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<RtpSenderInterface>,
              CreateSender,
              (const std::string&, const std::string&),
              (override));
  MOCK_METHOD(std::vector<rtc::scoped_refptr<RtpSenderInterface>>,
              GetSenders,
              (),
              (const, override));
  MOCK_METHOD(std::vector<rtc::scoped_refptr<RtpReceiverInterface>>,
              GetReceivers,
              (),
              (const, override));
  MOCK_METHOD(std::vector<rtc::scoped_refptr<RtpTransceiverInterface>>,
              GetTransceivers,
              (),
              (const, override));
  MOCK_METHOD(bool,
              GetStats,
              (StatsObserver*, MediaStreamTrackInterface*, StatsOutputLevel),
              (override));
  MOCK_METHOD(void, GetStats, (RTCStatsCollectorCallback*), (override));
  MOCK_METHOD(void,
              GetStats,
              (rtc::scoped_refptr<RtpSenderInterface>,
               rtc::scoped_refptr<RTCStatsCollectorCallback>),
              (override));
  MOCK_METHOD(void,
              GetStats,
              (rtc::scoped_refptr<RtpReceiverInterface>,
               rtc::scoped_refptr<RTCStatsCollectorCallback>),
              (override));
  MOCK_METHOD(void, ClearStatsCache, (), (override));
  MOCK_METHOD(rtc::scoped_refptr<SctpTransportInterface>,
              GetSctpTransport,
              (),
              (const, override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<DataChannelInterface>>,
              CreateDataChannelOrError,
              (const std::string&, const DataChannelInit*),
              (override));
  MOCK_METHOD(const SessionDescriptionInterface*,
              local_description,
              (),
              (const, override));
  MOCK_METHOD(const SessionDescriptionInterface*,
              remote_description,
              (),
              (const, override));
  MOCK_METHOD(const SessionDescriptionInterface*,
              current_local_description,
              (),
              (const, override));
  MOCK_METHOD(const SessionDescriptionInterface*,
              current_remote_description,
              (),
              (const, override));
  MOCK_METHOD(const SessionDescriptionInterface*,
              pending_local_description,
              (),
              (const, override));
  MOCK_METHOD(const SessionDescriptionInterface*,
              pending_remote_description,
              (),
              (const, override));
  MOCK_METHOD(void, RestartIce, (), (override));
  MOCK_METHOD(void,
              CreateOffer,
              (CreateSessionDescriptionObserver*, const RTCOfferAnswerOptions&),
              (override));
  MOCK_METHOD(void,
              CreateAnswer,
              (CreateSessionDescriptionObserver*, const RTCOfferAnswerOptions&),
              (override));
  MOCK_METHOD(void,
              SetLocalDescription,
              (SetSessionDescriptionObserver*, SessionDescriptionInterface*),
              (override));
  MOCK_METHOD(void,
              SetRemoteDescription,
              (SetSessionDescriptionObserver*, SessionDescriptionInterface*),
              (override));
  MOCK_METHOD(void,
              SetRemoteDescription,
              (std::unique_ptr<SessionDescriptionInterface>,
               rtc::scoped_refptr<SetRemoteDescriptionObserverInterface>),
              (override));
  MOCK_METHOD(bool,
              ShouldFireNegotiationNeededEvent,
              (uint32_t event_id),
              (override));
  MOCK_METHOD(PeerConnectionInterface::RTCConfiguration,
              GetConfiguration,
              (),
              (override));
  MOCK_METHOD(RTCError,
              SetConfiguration,
              (const PeerConnectionInterface::RTCConfiguration&),
              (override));
  MOCK_METHOD(bool,
              AddIceCandidate,
              (const IceCandidateInterface*),
              (override));
  MOCK_METHOD(bool,
              RemoveIceCandidates,
              (const std::vector<cricket::Candidate>&),
              (override));
  MOCK_METHOD(RTCError, SetBitrate, (const BitrateSettings&), (override));
  MOCK_METHOD(void,
              ReconfigureBandwidthEstimation,
              (const BandwidthEstimationSettings&),
              (override));
  MOCK_METHOD(void, SetAudioPlayout, (bool), (override));
  MOCK_METHOD(void, SetAudioRecording, (bool), (override));
  MOCK_METHOD(rtc::scoped_refptr<DtlsTransportInterface>,
              LookupDtlsTransportByMid,
              (const std::string&),
              (override));
  MOCK_METHOD(SignalingState, signaling_state, (), (override));
  MOCK_METHOD(IceConnectionState, ice_connection_state, (), (override));
  MOCK_METHOD(IceConnectionState,
              standardized_ice_connection_state,
              (),
              (override));
  MOCK_METHOD(PeerConnectionState, peer_connection_state, (), (override));
  MOCK_METHOD(IceGatheringState, ice_gathering_state, (), (override));
  MOCK_METHOD(void,
              AddAdaptationResource,
              (rtc::scoped_refptr<Resource>),
              (override));
  MOCK_METHOD(std::optional<bool>, can_trickle_ice_candidates, (), (override));
  MOCK_METHOD(bool,
              StartRtcEventLog,
              (std::unique_ptr<RtcEventLogOutput>, int64_t),
              (override));
  MOCK_METHOD(bool,
              StartRtcEventLog,
              (std::unique_ptr<RtcEventLogOutput>),
              (override));
  MOCK_METHOD(void, StopRtcEventLog, (), (override));
  MOCK_METHOD(void, Close, (), (override));
  MOCK_METHOD(rtc::Thread*, signaling_thread, (), (const, override));
  MOCK_METHOD(NetworkControllerInterface*,
              GetNetworkController,
              (),
              (override));
};

static_assert(
    !std::is_abstract_v<rtc::RefCountedObject<MockPeerConnectionInterface>>,
    "");

}  // namespace webrtc

#endif  // API_TEST_MOCK_PEERCONNECTIONINTERFACE_H_
