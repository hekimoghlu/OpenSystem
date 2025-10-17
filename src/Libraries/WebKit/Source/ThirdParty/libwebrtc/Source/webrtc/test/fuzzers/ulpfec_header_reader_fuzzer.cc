/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
#include <algorithm>

#include "api/scoped_refptr.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/forward_error_correction.h"
#include "modules/rtp_rtcp/source/ulpfec_header_reader_writer.h"

namespace webrtc {

using Packet = ForwardErrorCorrection::Packet;
using ReceivedFecPacket = ForwardErrorCorrection::ReceivedFecPacket;

void FuzzOneInput(const uint8_t* data, size_t size) {
  ReceivedFecPacket packet;
  packet.pkt = rtc::scoped_refptr<Packet>(new Packet());
  const size_t packet_size =
      std::min(size, static_cast<size_t>(IP_PACKET_SIZE));
  packet.pkt->data.SetSize(packet_size);
  packet.pkt->data.EnsureCapacity(IP_PACKET_SIZE);
  memcpy(packet.pkt->data.MutableData(), data, packet_size);

  UlpfecHeaderReader ulpfec_reader;
  ulpfec_reader.ReadFecHeader(&packet);
}

}  // namespace webrtc
