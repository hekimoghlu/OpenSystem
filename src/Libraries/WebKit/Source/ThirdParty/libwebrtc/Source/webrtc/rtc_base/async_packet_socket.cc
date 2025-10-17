/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#include "rtc_base/async_packet_socket.h"

#include "rtc_base/checks.h"

namespace rtc {

PacketTimeUpdateParams::PacketTimeUpdateParams() = default;

PacketTimeUpdateParams::PacketTimeUpdateParams(
    const PacketTimeUpdateParams& other) = default;

PacketTimeUpdateParams::~PacketTimeUpdateParams() = default;

PacketOptions::PacketOptions() = default;
PacketOptions::PacketOptions(DiffServCodePoint dscp) : dscp(dscp) {}
PacketOptions::PacketOptions(const PacketOptions& other) = default;
PacketOptions::~PacketOptions() = default;

AsyncPacketSocket::~AsyncPacketSocket() = default;

void AsyncPacketSocket::SubscribeCloseEvent(
    const void* removal_tag,
    std::function<void(AsyncPacketSocket*, int)> callback) {
  RTC_DCHECK_RUN_ON(&network_checker_);
  on_close_.AddReceiver(removal_tag, std::move(callback));
}

void AsyncPacketSocket::UnsubscribeCloseEvent(const void* removal_tag) {
  RTC_DCHECK_RUN_ON(&network_checker_);
  on_close_.RemoveReceivers(removal_tag);
}

void AsyncPacketSocket::RegisterReceivedPacketCallback(
    absl::AnyInvocable<void(AsyncPacketSocket*, const rtc::ReceivedPacket&)>
        received_packet_callback) {
  RTC_DCHECK_RUN_ON(&network_checker_);
  RTC_CHECK(!received_packet_callback_);
  received_packet_callback_ = std::move(received_packet_callback);
}

void AsyncPacketSocket::DeregisterReceivedPacketCallback() {
  RTC_DCHECK_RUN_ON(&network_checker_);
  received_packet_callback_ = nullptr;
}

void AsyncPacketSocket::NotifyPacketReceived(
    const rtc::ReceivedPacket& packet) {
  RTC_DCHECK_RUN_ON(&network_checker_);
  if (received_packet_callback_) {
    received_packet_callback_(this, packet);
    return;
  }
}

void CopySocketInformationToPacketInfo(size_t packet_size_bytes,
                                       const AsyncPacketSocket& socket_from,
                                       rtc::PacketInfo* info) {
  info->packet_size_bytes = packet_size_bytes;
  info->ip_overhead_bytes = socket_from.GetLocalAddress().ipaddr().overhead();
}

}  // namespace rtc
