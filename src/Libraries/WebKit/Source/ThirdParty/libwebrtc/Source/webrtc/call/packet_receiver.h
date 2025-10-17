/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#ifndef CALL_PACKET_RECEIVER_H_
#define CALL_PACKET_RECEIVER_H_

#include "absl/functional/any_invocable.h"
#include "api/media_types.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "rtc_base/copy_on_write_buffer.h"

namespace webrtc {

class PacketReceiver {
 public:
  // Demux RTCP packets. Must be called on the worker thread.
  virtual void DeliverRtcpPacket(rtc::CopyOnWriteBuffer packet) = 0;

  // Invoked once when a packet is received that can not be demuxed.
  // If the method returns true, a new attempt is made to demux the packet.
  using OnUndemuxablePacketHandler =
      absl::AnyInvocable<bool(const RtpPacketReceived& parsed_packet)>;

  // Must be called on the worker thread.
  // If `media_type` is not Audio or Video, packets may be used for BWE
  // calculations but are not demuxed.
  virtual void DeliverRtpPacket(
      MediaType media_type,
      RtpPacketReceived packet,
      OnUndemuxablePacketHandler undemuxable_packet_handler) = 0;

 protected:
  virtual ~PacketReceiver() {}
};

}  // namespace webrtc

#endif  // CALL_PACKET_RECEIVER_H_
