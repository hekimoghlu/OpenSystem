/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#include "p2p/base/packet_transport_internal.h"

#include "api/sequence_checker.h"
#include "rtc_base/network/received_packet.h"

namespace rtc {

PacketTransportInternal::PacketTransportInternal() = default;

PacketTransportInternal::~PacketTransportInternal() = default;

bool PacketTransportInternal::GetOption(rtc::Socket::Option opt, int* value) {
  return false;
}

std::optional<NetworkRoute> PacketTransportInternal::network_route() const {
  return std::optional<NetworkRoute>();
}

void PacketTransportInternal::RegisterReceivedPacketCallback(
    void* id,
    absl::AnyInvocable<void(PacketTransportInternal*,
                            const rtc::ReceivedPacket&)> callback) {
  RTC_DCHECK_RUN_ON(&network_checker_);
  received_packet_callback_list_.AddReceiver(id, std::move(callback));
}

void PacketTransportInternal::DeregisterReceivedPacketCallback(void* id) {
  RTC_DCHECK_RUN_ON(&network_checker_);
  received_packet_callback_list_.RemoveReceivers(id);
}

void PacketTransportInternal::SetOnCloseCallback(
    absl::AnyInvocable<void() &&> callback) {
  RTC_DCHECK_RUN_ON(&network_checker_);
  RTC_DCHECK(!on_close_ || !callback);
  on_close_ = std::move(callback);
}

void PacketTransportInternal::NotifyPacketReceived(
    const rtc::ReceivedPacket& packet) {
  RTC_DCHECK_RUN_ON(&network_checker_);
  received_packet_callback_list_.Send(this, packet);
}

void PacketTransportInternal::NotifyOnClose() {
  RTC_DCHECK_RUN_ON(&network_checker_);
  if (on_close_) {
    std::move(on_close_)();
    on_close_ = nullptr;
  }
}

}  // namespace rtc
