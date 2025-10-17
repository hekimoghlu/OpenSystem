/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
#include "test/network/traffic_route.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_minmax.h"

namespace webrtc {
namespace test {
namespace {

class NullReceiver : public EmulatedNetworkReceiverInterface {
 public:
  void OnPacketReceived(EmulatedIpPacket packet) override {}
};

class ActionReceiver : public EmulatedNetworkReceiverInterface {
 public:
  explicit ActionReceiver(std::function<void()> action) : action_(action) {}
  ~ActionReceiver() override = default;

  void OnPacketReceived(EmulatedIpPacket packet) override { action_(); }

 private:
  std::function<void()> action_;
};

}  // namespace

CrossTrafficRouteImpl::CrossTrafficRouteImpl(
    Clock* clock,
    EmulatedNetworkReceiverInterface* receiver,
    EmulatedEndpointImpl* endpoint)
    : clock_(clock), receiver_(receiver), endpoint_(endpoint) {
  null_receiver_ = std::make_unique<NullReceiver>();
  std::optional<uint16_t> port =
      endpoint_->BindReceiver(0, null_receiver_.get());
  RTC_DCHECK(port);
  null_receiver_port_ = port.value();
}
CrossTrafficRouteImpl::~CrossTrafficRouteImpl() = default;

void CrossTrafficRouteImpl::TriggerPacketBurst(size_t num_packets,
                                               size_t packet_size) {
  for (size_t i = 0; i < num_packets; ++i) {
    SendPacket(packet_size);
  }
}

void CrossTrafficRouteImpl::NetworkDelayedAction(size_t packet_size,
                                                 std::function<void()> action) {
  auto action_receiver = std::make_unique<ActionReceiver>(action);
  // BindOneShotReceiver arranges to free the port in the endpoint after the
  // action is done.
  std::optional<uint16_t> port =
      endpoint_->BindOneShotReceiver(0, action_receiver.get());
  RTC_DCHECK(port);
  actions_.push_back(std::move(action_receiver));
  SendPacket(packet_size, port.value());
}

void CrossTrafficRouteImpl::SendPacket(size_t packet_size) {
  SendPacket(packet_size, null_receiver_port_);
}

void CrossTrafficRouteImpl::SendPacket(size_t packet_size, uint16_t dest_port) {
  rtc::CopyOnWriteBuffer data(packet_size);
  std::fill_n(data.MutableData(), data.size(), 0);
  receiver_->OnPacketReceived(EmulatedIpPacket(
      /*from=*/rtc::SocketAddress(),
      rtc::SocketAddress(endpoint_->GetPeerLocalAddress(), dest_port), data,
      clock_->CurrentTime()));
}

}  // namespace test
}  // namespace webrtc
