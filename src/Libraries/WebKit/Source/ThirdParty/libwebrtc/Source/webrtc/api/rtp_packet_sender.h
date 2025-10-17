/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#ifndef API_RTP_PACKET_SENDER_H_
#define API_RTP_PACKET_SENDER_H_

#include <cstdint>
#include <memory>
#include <vector>

namespace webrtc {

class RtpPacketToSend;

class RtpPacketSender {
 public:
  virtual ~RtpPacketSender() = default;

  // Insert a set of packets into queue, for eventual transmission. Based on the
  // type of packets, they will be prioritized and scheduled relative to other
  // packets and the current target send rate.
  virtual void EnqueuePackets(
      std::vector<std::unique_ptr<RtpPacketToSend>> packets) = 0;

  // Clear any pending packets with the given SSRC from the queue.
  // TODO(crbug.com/1395081): Make pure virtual when downstream code has been
  // updated.
  virtual void RemovePacketsForSsrc(uint32_t /* ssrc */) {}
};

}  // namespace webrtc

#endif  // API_RTP_PACKET_SENDER_H_
