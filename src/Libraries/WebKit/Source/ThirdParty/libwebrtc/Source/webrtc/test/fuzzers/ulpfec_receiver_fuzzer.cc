/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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

#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/byte_io.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "modules/rtp_rtcp/source/ulpfec_receiver.h"
#include "test/fuzzers/fuzz_data_helper.h"

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

  uint32_t ulpfec_ssrc = ByteReader<uint32_t>::ReadLittleEndian(data + 0);
  uint16_t ulpfec_seq_num = ByteReader<uint16_t>::ReadLittleEndian(data + 4);
  uint32_t media_ssrc = ByteReader<uint32_t>::ReadLittleEndian(data + 6);
  uint16_t media_seq_num = ByteReader<uint16_t>::ReadLittleEndian(data + 10);

  DummyCallback callback;
  UlpfecReceiver receiver(ulpfec_ssrc, 0, &callback, Clock::GetRealTimeClock());

  test::FuzzDataHelper fuzz_data(rtc::MakeArrayView(data, size));
  while (fuzz_data.CanReadBytes(kMinDataNeeded)) {
    size_t packet_length = kRtpHeaderSize + fuzz_data.Read<uint8_t>();
    auto raw_packet = fuzz_data.ReadByteArray(packet_length);

    RtpPacketReceived parsed_packet;
    if (!parsed_packet.Parse(raw_packet))
      continue;

    // Overwrite the fields for the sequence number and SSRC with
    // consistent values for either a received UlpFEC packet or received media
    // packet. (We're still relying on libfuzzer to manage to generate packet
    // headers that interact together; this just ensures that we have two
    // consistent streams).
    if (fuzz_data.ReadOrDefaultValue<uint8_t>(0) % 2 == 0) {
      // Simulate UlpFEC packet.
      parsed_packet.SetSequenceNumber(ulpfec_seq_num++);
      parsed_packet.SetSsrc(ulpfec_ssrc);
    } else {
      // Simulate media packet.
      parsed_packet.SetSequenceNumber(media_seq_num++);
      parsed_packet.SetSsrc(media_ssrc);
    }

    receiver.AddReceivedRedPacket(parsed_packet);
  }

  receiver.ProcessReceivedFec();
}

}  // namespace webrtc
