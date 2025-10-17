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
#ifndef PC_TEST_MOCK_PEER_CONNECTION_INTERNAL_H_
#define PC_TEST_MOCK_PEER_CONNECTION_INTERNAL_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "api/audio/audio_device.h"
#include "pc/peer_connection_internal.h"
#include "test/gmock.h"

namespace webrtc {

class MockPeerConnectionInternal : public PeerConnectionInternal {
 public:
  MockPeerConnectionInternal() {}
  ~MockPeerConnectionInternal() = default;
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
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<DataChannelInterface>>,
              CreateDataChannelOrError,
              (const std::string&, const DataChannelInit*),
              (override));
  MOCK_METHOD(SessionDescriptionInterface*,
              local_description,
              (),
              (const, override));
  MOCK_METHOD(SessionDescriptionInterface*,
              remote_description,
              (),
              (const, override));
  MOCK_METHOD(SessionDescriptionInterface*,
              current_local_description,
              (),
              (const, override));
  MOCK_METHOD(SessionDescriptionInterface*,
              current_remote_description,
              (),
              (const, override));
  MOCK_METHOD(SessionDescriptionInterface*,
              pending_local_description,
              (),
              (const, override));
  MOCK_METHOD(SessionDescriptionInterface*,
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
  MOCK_METHOD(rtc::scoped_refptr<SctpTransportInterface>,
              GetSctpTransport,
              (),
              (const, override));
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

  // PeerConnectionSdpMethods
  MOCK_METHOD(std::string, session_id, (), (const, override));
  MOCK_METHOD(bool, NeedsIceRestart, (const std::string&), (const, override));
  MOCK_METHOD(std::optional<std::string>, sctp_mid, (), (const, override));
  MOCK_METHOD(PeerConnectionInterface::RTCConfiguration*,
              configuration,
              (),
              (const, override));
  MOCK_METHOD(void,
              ReportSdpBundleUsage,
              (const SessionDescriptionInterface&),
              (override));
  MOCK_METHOD(PeerConnectionMessageHandler*, message_handler, (), (override));
  MOCK_METHOD(RtpTransmissionManager*, rtp_manager, (), (override));
  MOCK_METHOD(const RtpTransmissionManager*,
              rtp_manager,
              (),
              (const, override));
  MOCK_METHOD(bool, dtls_enabled, (), (const, override));
  MOCK_METHOD(const PeerConnectionFactoryInterface::Options*,
              options,
              (),
              (const, override));
  MOCK_METHOD(CryptoOptions, GetCryptoOptions, (), (override));
  MOCK_METHOD(JsepTransportController*, transport_controller_s, (), (override));
  MOCK_METHOD(JsepTransportController*, transport_controller_n, (), (override));
  MOCK_METHOD(DataChannelController*, data_channel_controller, (), (override));
  MOCK_METHOD(cricket::PortAllocator*, port_allocator, (), (override));
  MOCK_METHOD(LegacyStatsCollector*, legacy_stats, (), (override));
  MOCK_METHOD(PeerConnectionObserver*, Observer, (), (const, override));
  MOCK_METHOD(std::optional<rtc::SSLRole>, GetSctpSslRole_n, (), (override));
  MOCK_METHOD(PeerConnectionInterface::IceConnectionState,
              ice_connection_state_internal,
              (),
              (override));
  MOCK_METHOD(void,
              SetIceConnectionState,
              (PeerConnectionInterface::IceConnectionState),
              (override));
  MOCK_METHOD(void, NoteUsageEvent, (UsageEvent), (override));
  MOCK_METHOD(bool, IsClosed, (), (const, override));
  MOCK_METHOD(bool, IsUnifiedPlan, (), (const, override));
  MOCK_METHOD(bool,
              ValidateBundleSettings,
              (const cricket::SessionDescription*,
               (const std::map<std::string, const cricket::ContentGroup*>&)),
              (override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              (cricket::MediaType,
               rtc::scoped_refptr<MediaStreamTrackInterface>,
               const RtpTransceiverInit&,
               bool),
              (override));
  MOCK_METHOD(void, StartSctpTransport, (int, int, int), (override));
  MOCK_METHOD(void,
              AddRemoteCandidate,
              (const std::string&, const cricket::Candidate&),
              (override));
  MOCK_METHOD(Call*, call_ptr, (), (override));
  MOCK_METHOD(bool, SrtpRequired, (), (const, override));
  MOCK_METHOD(bool,
              CreateDataChannelTransport,
              (absl::string_view),
              (override));
  MOCK_METHOD(void, DestroyDataChannelTransport, (RTCError error), (override));
  MOCK_METHOD(const FieldTrialsView&, trials, (), (const, override));

  // PeerConnectionInternal
  MOCK_METHOD(rtc::Thread*, network_thread, (), (const, override));
  MOCK_METHOD(rtc::Thread*, worker_thread, (), (const, override));
  MOCK_METHOD(bool, initial_offerer, (), (const, override));
  MOCK_METHOD(
      std::vector<
          rtc::scoped_refptr<RtpTransceiverProxyWithInternal<RtpTransceiver>>>,
      GetTransceiversInternal,
      (),
      (const, override));
  MOCK_METHOD(std::vector<DataChannelStats>,
              GetDataChannelStats,
              (),
              (const, override));
  MOCK_METHOD(std::optional<std::string>,
              sctp_transport_name,
              (),
              (const, override));
  MOCK_METHOD(cricket::CandidateStatsList,
              GetPooledCandidateStats,
              (),
              (const, override));
  MOCK_METHOD((std::map<std::string, cricket::TransportStats>),
              GetTransportStatsByNames,
              (const std::set<std::string>&),
              (override));
  MOCK_METHOD(Call::Stats, GetCallStats, (), (override));
  MOCK_METHOD(std::optional<AudioDeviceModule::Stats>,
              GetAudioDeviceStats,
              (),
              (override));
  MOCK_METHOD(bool,
              GetLocalCertificate,
              (const std::string&, rtc::scoped_refptr<rtc::RTCCertificate>*),
              (override));
  MOCK_METHOD(std::unique_ptr<rtc::SSLCertChain>,
              GetRemoteSSLCertChain,
              (const std::string&),
              (override));
  MOCK_METHOD(bool, IceRestartPending, (const std::string&), (const, override));
  MOCK_METHOD(bool,
              GetSslRole,
              (const std::string&, rtc::SSLRole*),
              (override));
  MOCK_METHOD(void, NoteDataAddedEvent, (), (override));
  MOCK_METHOD(void,
              OnSctpDataChannelStateChanged,
              (int channel_id, DataChannelInterface::DataState),
              (override));
  MOCK_METHOD(NetworkControllerInterface*,
              GetNetworkController,
              (),
              (override));
  MOCK_METHOD(PayloadTypePicker&, payload_type_picker, (), (override));
};

}  // namespace webrtc

#endif  // PC_TEST_MOCK_PEER_CONNECTION_INTERNAL_H_
