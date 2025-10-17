/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#ifndef TEST_PEER_SCENARIO_SCENARIO_CONNECTION_H_
#define TEST_PEER_SCENARIO_SCENARIO_CONNECTION_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "api/candidate.h"
#include "api/environment/environment.h"
#include "api/jsep.h"
#include "p2p/base/transport_description.h"
#include "test/network/network_emulation_manager.h"
#include "test/scoped_key_value_config.h"

namespace webrtc {

// ScenarioIceConnection provides the transport level functionality of a
// PeerConnection for use in peer connection scenario tests. This allows
// implementing custom server side behavior in tests.
class ScenarioIceConnection {
 public:
  class IceConnectionObserver {
   public:
    // Called on network thread.
    virtual void OnPacketReceived(rtc::CopyOnWriteBuffer packet) = 0;
    // Called on signaling thread.
    virtual void OnIceCandidates(
        const std::string& mid,
        const std::vector<cricket::Candidate>& candidates) = 0;

   protected:
    ~IceConnectionObserver() = default;
  };
  static std::unique_ptr<ScenarioIceConnection> Create(
      const Environment& env,
      test::NetworkEmulationManagerImpl* net,
      IceConnectionObserver* observer);

  virtual ~ScenarioIceConnection() = default;

  // Posts tasks to send packets to network thread.
  virtual void SendRtpPacket(rtc::ArrayView<const uint8_t> packet_view) = 0;
  virtual void SendRtcpPacket(rtc::ArrayView<const uint8_t> packet_view) = 0;

  // Used for ICE configuration, called on signaling thread.
  virtual void SetRemoteSdp(SdpType type, const std::string& remote_sdp) = 0;
  virtual void SetLocalSdp(SdpType type, const std::string& local_sdp) = 0;

  virtual EmulatedEndpoint* endpoint() = 0;
  virtual const cricket::TransportDescription& transport_description()
      const = 0;

  webrtc::test::ScopedKeyValueConfig field_trials;
};

}  // namespace webrtc

#endif  // TEST_PEER_SCENARIO_SCENARIO_CONNECTION_H_
