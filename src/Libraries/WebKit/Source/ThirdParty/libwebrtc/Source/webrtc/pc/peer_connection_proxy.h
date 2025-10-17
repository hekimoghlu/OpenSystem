/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#ifndef PC_PEER_CONNECTION_PROXY_H_
#define PC_PEER_CONNECTION_PROXY_H_

#include <memory>
#include <string>
#include <vector>

#include "api/peer_connection_interface.h"
#include "api/transport/bandwidth_estimation_settings.h"
#include "pc/proxy.h"

namespace webrtc {

// PeerConnection proxy objects will be constructed with two thread pointers,
// signaling and network. The proxy macros don't have 'network' specific macros
// and support for a secondary thread is provided via 'SECONDARY' macros.
// TODO(deadbeef): Move this to .cc file. What threads methods are called on is
// an implementation detail.
BEGIN_PROXY_MAP(PeerConnection)
PROXY_PRIMARY_THREAD_DESTRUCTOR()
PROXY_METHOD0(rtc::scoped_refptr<StreamCollectionInterface>, local_streams)
PROXY_METHOD0(rtc::scoped_refptr<StreamCollectionInterface>, remote_streams)
PROXY_METHOD1(bool, AddStream, MediaStreamInterface*)
PROXY_METHOD1(void, RemoveStream, MediaStreamInterface*)
PROXY_METHOD2(RTCErrorOr<rtc::scoped_refptr<RtpSenderInterface>>,
              AddTrack,
              rtc::scoped_refptr<MediaStreamTrackInterface>,
              const std::vector<std::string>&)
PROXY_METHOD3(RTCErrorOr<rtc::scoped_refptr<RtpSenderInterface>>,
              AddTrack,
              rtc::scoped_refptr<MediaStreamTrackInterface>,
              const std::vector<std::string>&,
              const std::vector<RtpEncodingParameters>&)
PROXY_METHOD1(RTCError,
              RemoveTrackOrError,
              rtc::scoped_refptr<RtpSenderInterface>)
PROXY_METHOD1(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              rtc::scoped_refptr<MediaStreamTrackInterface>)
PROXY_METHOD2(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              rtc::scoped_refptr<MediaStreamTrackInterface>,
              const RtpTransceiverInit&)
PROXY_METHOD1(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              cricket::MediaType)
PROXY_METHOD2(RTCErrorOr<rtc::scoped_refptr<RtpTransceiverInterface>>,
              AddTransceiver,
              cricket::MediaType,
              const RtpTransceiverInit&)
PROXY_METHOD2(rtc::scoped_refptr<RtpSenderInterface>,
              CreateSender,
              const std::string&,
              const std::string&)
PROXY_CONSTMETHOD0(std::vector<rtc::scoped_refptr<RtpSenderInterface>>,
                   GetSenders)
PROXY_CONSTMETHOD0(std::vector<rtc::scoped_refptr<RtpReceiverInterface>>,
                   GetReceivers)
PROXY_CONSTMETHOD0(std::vector<rtc::scoped_refptr<RtpTransceiverInterface>>,
                   GetTransceivers)
PROXY_METHOD3(bool,
              GetStats,
              StatsObserver*,
              MediaStreamTrackInterface*,
              StatsOutputLevel)
PROXY_METHOD1(void, GetStats, RTCStatsCollectorCallback*)
PROXY_METHOD2(void,
              GetStats,
              rtc::scoped_refptr<RtpSenderInterface>,
              rtc::scoped_refptr<RTCStatsCollectorCallback>)
PROXY_METHOD2(void,
              GetStats,
              rtc::scoped_refptr<RtpReceiverInterface>,
              rtc::scoped_refptr<RTCStatsCollectorCallback>)
PROXY_METHOD0(void, ClearStatsCache)
PROXY_METHOD2(RTCErrorOr<rtc::scoped_refptr<DataChannelInterface>>,
              CreateDataChannelOrError,
              const std::string&,
              const DataChannelInit*)
PROXY_CONSTMETHOD0(const SessionDescriptionInterface*, local_description)
PROXY_CONSTMETHOD0(const SessionDescriptionInterface*, remote_description)
PROXY_CONSTMETHOD0(const SessionDescriptionInterface*,
                   current_local_description)
PROXY_CONSTMETHOD0(const SessionDescriptionInterface*,
                   current_remote_description)
PROXY_CONSTMETHOD0(const SessionDescriptionInterface*,
                   pending_local_description)
PROXY_CONSTMETHOD0(const SessionDescriptionInterface*,
                   pending_remote_description)
PROXY_METHOD0(void, RestartIce)
PROXY_METHOD2(void,
              CreateOffer,
              CreateSessionDescriptionObserver*,
              const RTCOfferAnswerOptions&)
PROXY_METHOD2(void,
              CreateAnswer,
              CreateSessionDescriptionObserver*,
              const RTCOfferAnswerOptions&)
PROXY_METHOD2(void,
              SetLocalDescription,
              std::unique_ptr<SessionDescriptionInterface>,
              rtc::scoped_refptr<SetLocalDescriptionObserverInterface>)
PROXY_METHOD1(void,
              SetLocalDescription,
              rtc::scoped_refptr<SetLocalDescriptionObserverInterface>)
PROXY_METHOD2(void,
              SetLocalDescription,
              SetSessionDescriptionObserver*,
              SessionDescriptionInterface*)
PROXY_METHOD1(void, SetLocalDescription, SetSessionDescriptionObserver*)
PROXY_METHOD2(void,
              SetRemoteDescription,
              std::unique_ptr<SessionDescriptionInterface>,
              rtc::scoped_refptr<SetRemoteDescriptionObserverInterface>)
PROXY_METHOD2(void,
              SetRemoteDescription,
              SetSessionDescriptionObserver*,
              SessionDescriptionInterface*)
PROXY_METHOD1(bool, ShouldFireNegotiationNeededEvent, uint32_t)
PROXY_METHOD0(PeerConnectionInterface::RTCConfiguration, GetConfiguration)
PROXY_METHOD1(RTCError,
              SetConfiguration,
              const PeerConnectionInterface::RTCConfiguration&)
PROXY_METHOD1(bool, AddIceCandidate, const IceCandidateInterface*)
PROXY_METHOD2(void,
              AddIceCandidate,
              std::unique_ptr<IceCandidateInterface>,
              std::function<void(RTCError)>)
PROXY_METHOD1(bool, RemoveIceCandidates, const std::vector<cricket::Candidate>&)
PROXY_METHOD1(RTCError, SetBitrate, const BitrateSettings&)
PROXY_METHOD1(void,
              ReconfigureBandwidthEstimation,
              const BandwidthEstimationSettings&)
PROXY_METHOD1(void, SetAudioPlayout, bool)
PROXY_METHOD1(void, SetAudioRecording, bool)
// This method will be invoked on the network thread. See
// PeerConnectionFactory::CreatePeerConnectionOrError for more details.
PROXY_SECONDARY_METHOD1(rtc::scoped_refptr<DtlsTransportInterface>,
                        LookupDtlsTransportByMid,
                        const std::string&)
// This method will be invoked on the network thread. See
// PeerConnectionFactory::CreatePeerConnectionOrError for more details.
PROXY_SECONDARY_CONSTMETHOD0(rtc::scoped_refptr<SctpTransportInterface>,
                             GetSctpTransport)
PROXY_METHOD0(SignalingState, signaling_state)
PROXY_METHOD0(IceConnectionState, ice_connection_state)
PROXY_METHOD0(IceConnectionState, standardized_ice_connection_state)
PROXY_METHOD0(PeerConnectionState, peer_connection_state)
PROXY_METHOD0(IceGatheringState, ice_gathering_state)
PROXY_METHOD0(std::optional<bool>, can_trickle_ice_candidates)
PROXY_METHOD1(void, AddAdaptationResource, rtc::scoped_refptr<Resource>)
PROXY_METHOD2(bool,
              StartRtcEventLog,
              std::unique_ptr<RtcEventLogOutput>,
              int64_t)
PROXY_METHOD1(bool, StartRtcEventLog, std::unique_ptr<RtcEventLogOutput>)
PROXY_METHOD0(void, StopRtcEventLog)
PROXY_METHOD0(void, Close)
PROXY_METHOD0(NetworkControllerInterface*, GetNetworkController)
BYPASS_PROXY_CONSTMETHOD0(rtc::Thread*, signaling_thread)
END_PROXY_MAP(PeerConnection)

}  // namespace webrtc

#endif  // PC_PEER_CONNECTION_PROXY_H_
