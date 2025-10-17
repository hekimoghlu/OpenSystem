/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#ifndef RTC_TOOLS_DATA_CHANNEL_BENCHMARK_PEER_CONNECTION_CLIENT_H_
#define RTC_TOOLS_DATA_CHANNEL_BENCHMARK_PEER_CONNECTION_CLIENT_H_

#include <stdint.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "api/data_channel_interface.h"
#include "api/jsep.h"
#include "api/peer_connection_interface.h"
#include "api/rtp_receiver_interface.h"
#include "api/scoped_refptr.h"
#include "api/set_local_description_observer_interface.h"
#include "rtc_base/logging.h"
#include "rtc_base/thread.h"
#include "rtc_tools/data_channel_benchmark/signaling_interface.h"

namespace webrtc {

// Handles all the details for creating a PeerConnection and negotiation using a
// SignalingInterface object.
class PeerConnectionClient : public webrtc::PeerConnectionObserver {
 public:
  explicit PeerConnectionClient(webrtc::PeerConnectionFactoryInterface* factory,
                                webrtc::SignalingInterface* signaling);

  ~PeerConnectionClient() override;

  PeerConnectionClient(const PeerConnectionClient&) = delete;
  PeerConnectionClient& operator=(const PeerConnectionClient&) = delete;

  // Set the local description and send offer using the SignalingInterface,
  // initiating the negotiation process.
  bool StartPeerConnection();

  // Whether the peer connection is connected to the remote peer.
  bool IsConnected();

  // Disconnect from the call.
  void Disconnect();

  rtc::scoped_refptr<webrtc::PeerConnectionInterface> peerConnection() {
    return peer_connection_;
  }

  // Set a callback to run when a DataChannel is created by the remote peer.
  void SetOnDataChannel(
      std::function<void(rtc::scoped_refptr<webrtc::DataChannelInterface>)>
          callback);

  std::vector<rtc::scoped_refptr<webrtc::DataChannelInterface>>&
  dataChannels() {
    return data_channels_;
  }

  // Creates a default PeerConnectionFactory object.
  static rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface>
  CreateDefaultFactory(rtc::Thread* signaling_thread);

 private:
  void AddIceCandidate(
      std::unique_ptr<webrtc::IceCandidateInterface> candidate);
  bool SetRemoteDescription(
      std::unique_ptr<webrtc::SessionDescriptionInterface> desc);

  // Initialize the PeerConnection with a given PeerConnectionFactory.
  bool InitializePeerConnection(
      webrtc::PeerConnectionFactoryInterface* factory);
  void DeletePeerConnection();

  // PeerConnectionObserver implementation.
  void OnSignalingChange(
      webrtc::PeerConnectionInterface::SignalingState new_state) override {
    RTC_LOG(LS_INFO) << __FUNCTION__ << " new state: " << new_state;
  }
  void OnDataChannel(
      rtc::scoped_refptr<webrtc::DataChannelInterface> channel) override;
  void OnNegotiationNeededEvent(uint32_t event_id) override;
  void OnIceConnectionChange(
      webrtc::PeerConnectionInterface::IceConnectionState new_state) override;
  void OnIceGatheringChange(
      webrtc::PeerConnectionInterface::IceGatheringState new_state) override;
  void OnIceCandidate(const webrtc::IceCandidateInterface* candidate) override;
  void OnIceConnectionReceivingChange(bool receiving) override {
    RTC_LOG(LS_INFO) << __FUNCTION__ << " receiving? " << receiving;
  }

  rtc::scoped_refptr<webrtc::PeerConnectionInterface> peer_connection_;
  std::function<void(rtc::scoped_refptr<webrtc::DataChannelInterface>)>
      on_data_channel_callback_;
  std::vector<rtc::scoped_refptr<webrtc::DataChannelInterface>> data_channels_;
  webrtc::SignalingInterface* signaling_;
};

}  // namespace webrtc

#endif  // RTC_TOOLS_DATA_CHANNEL_BENCHMARK_PEER_CONNECTION_CLIENT_H_
