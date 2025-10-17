/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
#ifndef NET_DCSCTP_SOCKET_PACKET_SENDER_H_
#define NET_DCSCTP_SOCKET_PACKET_SENDER_H_

#include "net/dcsctp/packet/sctp_packet.h"
#include "net/dcsctp/public/dcsctp_socket.h"

namespace dcsctp {

// The PacketSender sends packets to the network using the provided callback
// interface. When an attempt to send a packet is made, the `on_sent_packet`
// callback will be triggered.
class PacketSender {
 public:
  PacketSender(DcSctpSocketCallbacks& callbacks,
               std::function<void(rtc::ArrayView<const uint8_t>,
                                  SendPacketStatus)> on_sent_packet);

  // Sends the packet, and returns true if it was sent successfully.
  bool Send(SctpPacket::Builder& builder, bool write_checksum = true);

 private:
  DcSctpSocketCallbacks& callbacks_;

  // Callback that will be triggered for every send attempt, indicating the
  // status of the operation.
  std::function<void(rtc::ArrayView<const uint8_t>, SendPacketStatus)>
      on_sent_packet_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_SOCKET_PACKET_SENDER_H_
