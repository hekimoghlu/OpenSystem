/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_PACKET_SOURCE_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_PACKET_SOURCE_H_

#include <bitset>
#include <memory>

#include "modules/audio_coding/neteq/tools/packet.h"

namespace webrtc {
namespace test {

// Interface class for an object delivering RTP packets to test applications.
class PacketSource {
 public:
  PacketSource();
  virtual ~PacketSource();

  PacketSource(const PacketSource&) = delete;
  PacketSource& operator=(const PacketSource&) = delete;

  // Returns next packet. Returns nullptr if the source is depleted, or if an
  // error occurred.
  virtual std::unique_ptr<Packet> NextPacket() = 0;

  virtual void FilterOutPayloadType(uint8_t payload_type);

 protected:
  std::bitset<128> filter_;  // Payload type is 7 bits in the RFC.
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_PACKET_SOURCE_H_
