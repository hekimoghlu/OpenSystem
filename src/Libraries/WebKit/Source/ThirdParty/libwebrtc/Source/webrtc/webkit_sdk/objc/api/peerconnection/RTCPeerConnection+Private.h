/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
#import "RTCPeerConnection.h"

#include "api/peer_connection_interface.h"

NS_ASSUME_NONNULL_BEGIN

namespace webrtc {

/**
 * These objects are created by RTCPeerConnectionFactory to wrap an
 * id<RTCPeerConnectionDelegate> and call methods on that interface.
 */
class PeerConnectionDelegateAdapter : public PeerConnectionObserver {
 public:
  PeerConnectionDelegateAdapter(RTCPeerConnection *peerConnection);
  ~PeerConnectionDelegateAdapter() override;

  void OnSignalingChange(PeerConnectionInterface::SignalingState new_state) override;

  void OnAddStream(rtc::scoped_refptr<MediaStreamInterface> stream) override;

  void OnRemoveStream(rtc::scoped_refptr<MediaStreamInterface> stream) override;

  void OnTrack(rtc::scoped_refptr<RtpTransceiverInterface> transceiver) override;

  void OnDataChannel(rtc::scoped_refptr<DataChannelInterface> data_channel) override;

  void OnRenegotiationNeeded() override;

  void OnIceConnectionChange(PeerConnectionInterface::IceConnectionState new_state) override;

  void OnStandardizedIceConnectionChange(
      PeerConnectionInterface::IceConnectionState new_state) override;

  void OnConnectionChange(PeerConnectionInterface::PeerConnectionState new_state) override;

  void OnIceGatheringChange(PeerConnectionInterface::IceGatheringState new_state) override;

  void OnIceCandidate(const IceCandidateInterface *candidate) override;

  void OnIceCandidatesRemoved(const std::vector<cricket::Candidate> &candidates) override;

  void OnIceSelectedCandidatePairChanged(const cricket::CandidatePairChangeEvent &event) override;

  void OnAddTrack(rtc::scoped_refptr<RtpReceiverInterface> receiver,
                  const std::vector<rtc::scoped_refptr<MediaStreamInterface>> &streams) override;

  void OnRemoveTrack(rtc::scoped_refptr<RtpReceiverInterface> receiver) override;

 private:
  __weak RTCPeerConnection *peer_connection_;
};

}  // namespace webrtc

@interface RTCPeerConnection ()

/** The factory used to create this RTCPeerConnection */
@property(nonatomic, readonly) RTCPeerConnectionFactory *factory;

/** The native PeerConnectionInterface created during construction. */
@property(nonatomic, readonly) rtc::scoped_refptr<webrtc::PeerConnectionInterface>
    nativePeerConnection;

/** Initialize an RTCPeerConnection with a configuration, constraints, and
 *  delegate.
 */
- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                  configuration:(RTCConfiguration *)configuration
                    constraints:(RTCMediaConstraints *)constraints
                       delegate:(nullable id<RTCPeerConnectionDelegate>)delegate;

/** Initialize an RTCPeerConnection with a configuration, constraints,
 *  delegate and PeerConnectionDependencies.
 */
- (instancetype)initWithDependencies:(RTCPeerConnectionFactory *)factory
                       configuration:(RTCConfiguration *)configuration
                         constraints:(RTCMediaConstraints *)constraints
                        dependencies:
                            (std::unique_ptr<webrtc::PeerConnectionDependencies>)dependencies
                            delegate:(nullable id<RTCPeerConnectionDelegate>)delegate
    NS_DESIGNATED_INITIALIZER;

+ (webrtc::PeerConnectionInterface::SignalingState)nativeSignalingStateForState:
        (RTCSignalingState)state;

+ (RTCSignalingState)signalingStateForNativeState:
        (webrtc::PeerConnectionInterface::SignalingState)nativeState;

+ (NSString *)stringForSignalingState:(RTCSignalingState)state;

+ (webrtc::PeerConnectionInterface::IceConnectionState)nativeIceConnectionStateForState:
        (RTCIceConnectionState)state;

+ (webrtc::PeerConnectionInterface::PeerConnectionState)nativeConnectionStateForState:
        (RTCPeerConnectionState)state;

+ (RTCIceConnectionState)iceConnectionStateForNativeState:
        (webrtc::PeerConnectionInterface::IceConnectionState)nativeState;

+ (RTCPeerConnectionState)connectionStateForNativeState:
        (webrtc::PeerConnectionInterface::PeerConnectionState)nativeState;

+ (NSString *)stringForIceConnectionState:(RTCIceConnectionState)state;

+ (NSString *)stringForConnectionState:(RTCPeerConnectionState)state;

+ (webrtc::PeerConnectionInterface::IceGatheringState)nativeIceGatheringStateForState:
        (RTCIceGatheringState)state;

+ (RTCIceGatheringState)iceGatheringStateForNativeState:
        (webrtc::PeerConnectionInterface::IceGatheringState)nativeState;

+ (NSString *)stringForIceGatheringState:(RTCIceGatheringState)state;

+ (webrtc::PeerConnectionInterface::StatsOutputLevel)nativeStatsOutputLevelForLevel:
        (RTCStatsOutputLevel)level;

@end

NS_ASSUME_NONNULL_END
