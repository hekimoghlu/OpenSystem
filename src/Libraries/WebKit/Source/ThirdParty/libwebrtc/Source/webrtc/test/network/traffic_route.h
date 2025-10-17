/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#ifndef TEST_NETWORK_TRAFFIC_ROUTE_H_
#define TEST_NETWORK_TRAFFIC_ROUTE_H_

#include <memory>
#include <vector>

#include "api/test/network_emulation_manager.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "system_wrappers/include/clock.h"
#include "test/network/network_emulation.h"

namespace webrtc {
namespace test {

// Represents the endpoint for cross traffic that is going through the network.
// It can be used to emulate unexpected network load.
class CrossTrafficRouteImpl final : public CrossTrafficRoute {
 public:
  CrossTrafficRouteImpl(Clock* clock,
                        EmulatedNetworkReceiverInterface* receiver,
                        EmulatedEndpointImpl* endpoint);
  ~CrossTrafficRouteImpl();

  // Triggers sending of dummy packets with size `packet_size` bytes.
  void TriggerPacketBurst(size_t num_packets, size_t packet_size) override;
  // Sends a packet over the nodes and runs `action` when it has been delivered.
  void NetworkDelayedAction(size_t packet_size,
                            std::function<void()> action) override;

  void SendPacket(size_t packet_size) override;

 private:
  void SendPacket(size_t packet_size, uint16_t dest_port);

  Clock* const clock_;
  EmulatedNetworkReceiverInterface* const receiver_;
  EmulatedEndpointImpl* const endpoint_;

  uint16_t null_receiver_port_;
  std::unique_ptr<EmulatedNetworkReceiverInterface> null_receiver_;
  std::vector<std::unique_ptr<EmulatedNetworkReceiverInterface>> actions_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_NETWORK_TRAFFIC_ROUTE_H_
