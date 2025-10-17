/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#ifndef TEST_NETWORK_FEEDBACK_GENERATOR_H_
#define TEST_NETWORK_FEEDBACK_GENERATOR_H_

#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "api/transport/network_types.h"
#include "api/transport/test/feedback_generator_interface.h"
#include "test/network/network_emulation.h"
#include "test/network/network_emulation_manager.h"
#include "test/network/simulated_network.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {

class FeedbackGeneratorImpl
    : public FeedbackGenerator,
      public TwoWayFakeTrafficRoute<SentPacket, std::vector<PacketResult>>::
          TrafficHandlerInterface {
 public:
  explicit FeedbackGeneratorImpl(Config config);
  Timestamp Now() override;
  void Sleep(TimeDelta duration) override;
  void SendPacket(size_t size) override;
  std::vector<TransportPacketsFeedback> PopFeedback() override;

  void SetSendConfig(BuiltInNetworkBehaviorConfig config) override;
  void SetReturnConfig(BuiltInNetworkBehaviorConfig config) override;

  void SetSendLinkCapacity(DataRate capacity) override;

  void OnRequest(SentPacket packet, Timestamp arrival_time) override;
  void OnResponse(std::vector<PacketResult> packet_results,
                  Timestamp arrival_time) override;

 private:
  Config conf_;
  ::webrtc::test::NetworkEmulationManagerImpl net_;
  SimulatedNetwork* const send_link_;
  SimulatedNetwork* const ret_link_;
  TwoWayFakeTrafficRoute<SentPacket, std::vector<PacketResult>> route_;

  std::queue<SentPacket> sent_packets_;
  std::vector<PacketResult> received_packets_;
  std::vector<TransportPacketsFeedback> feedback_;
  int64_t sequence_number_ = 1;
};
}  // namespace webrtc
#endif  // TEST_NETWORK_FEEDBACK_GENERATOR_H_
