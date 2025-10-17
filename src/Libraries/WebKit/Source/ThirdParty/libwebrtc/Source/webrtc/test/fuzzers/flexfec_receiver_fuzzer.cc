/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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

#include "modules/rtp_rtcp/include/flexfec_receiver.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/byte_io.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"

namespace webrtc {

namespace {
class DummyCallback : public RecoveredPacketReceiver {
  void OnRecoveredPacket(const RtpPacketReceived& packet) override {}
};
}  // namespace

void FuzzOneInput(const uint8_t* data, size_t size) {
  constexpr size_t kMinDataNeeded = 12;
  if (size < kMinDataNeeded || size > 2000) {
    return;
  }

  uint32_t flexfec_ssrc;
  memcpy(&flexfec_ssrc, data + 0, 4);
  uint16_t flexfec_seq_num;
  memcpy(&flexfec_seq_num, data + 4, 2);
  uint32_t media_ssrc;
  memcpy(&media_ssrc, data + 6, 4);
  uint16_t media_seq_num;
  memcpy(&media_seq_num, data + 10, 2);

  DummyCallback callback;
  FlexfecReceiver receiver(flexfec_ssrc, media_ssrc, &callback);

  std::unique_ptr<uint8_t[]> packet;
  size_t packet_length;
  size_t i = kMinDataNeeded;
  while (i < size) {
    packet_length = kRtpHeaderSize + data[i++];
    packet = std::unique_ptr<uint8_t[]>(new uint8_t[packet_length]);
    if (i + packet_length >= size) {
      break;
    }
    memcpy(packet.get(), data + i, packet_length);
    i += packet_length;
    if (i < size && data[i++] % 2 == 0) {
      // Simulate FlexFEC packet.
      ByteWriter<uint16_t>::WriteBigEndian(packet.get() + 2, flexfec_seq_num++);
      ByteWriter<uint32_t>::WriteBigEndian(packet.get() + 8, flexfec_ssrc);
    } else {
      // Simulate media packet.
      ByteWriter<uint16_t>::WriteBigEndian(packet.get() + 2, media_seq_num++);
      ByteWriter<uint32_t>::WriteBigEndian(packet.get() + 8, media_ssrc);
    }
    RtpPacketReceived parsed_packet;
    if (parsed_packet.Parse(packet.get(), packet_length)) {
      receiver.OnRtpPacket(parsed_packet);
    }
  }
}

}  // namespace webrtc
