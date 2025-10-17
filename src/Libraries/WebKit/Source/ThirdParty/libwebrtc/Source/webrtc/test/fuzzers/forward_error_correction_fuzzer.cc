/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#include <memory>

#include "api/scoped_refptr.h"
#include "modules/rtp_rtcp/source/byte_io.h"
#include "modules/rtp_rtcp/source/forward_error_correction.h"
#include "rtc_base/byte_buffer.h"

namespace webrtc {

namespace {
constexpr uint32_t kMediaSsrc = 100200300;
constexpr uint32_t kFecSsrc = 111222333;

constexpr size_t kPacketSize = 50;
constexpr size_t kMaxPacketsInBuffer = 48;
}  // namespace

void FuzzOneInput(const uint8_t* data, size_t size) {
  if (size > 5000) {
    return;
  }
  // Object under test.
  std::unique_ptr<ForwardErrorCorrection> fec =
      ForwardErrorCorrection::CreateFlexfec(kFecSsrc, kMediaSsrc);

  // Entropy from fuzzer.
  rtc::ByteBufferReader fuzz_buffer(rtc::MakeArrayView(data, size));

  // Initial stream state.
  uint16_t media_seqnum;
  if (!fuzz_buffer.ReadUInt16(&media_seqnum))
    return;
  const uint16_t original_media_seqnum = media_seqnum;
  uint16_t fec_seqnum;
  if (!fuzz_buffer.ReadUInt16(&fec_seqnum))
    return;

  // Existing packets in the packet buffer.
  ForwardErrorCorrection::RecoveredPacketList recovered_packets;
  uint8_t num_existing_recovered_packets;
  if (!fuzz_buffer.ReadUInt8(&num_existing_recovered_packets))
    return;
  for (size_t i = 0; i < num_existing_recovered_packets % kMaxPacketsInBuffer;
       ++i) {
    ForwardErrorCorrection::RecoveredPacket* recovered_packet =
        new ForwardErrorCorrection::RecoveredPacket();
    recovered_packet->pkt = rtc::scoped_refptr<ForwardErrorCorrection::Packet>(
        new ForwardErrorCorrection::Packet());
    recovered_packet->pkt->data.SetSize(kPacketSize);
    memset(recovered_packet->pkt->data.MutableData(), 0, kPacketSize);
    recovered_packet->ssrc = kMediaSsrc;
    recovered_packet->seq_num = media_seqnum++;
    recovered_packets.emplace_back(recovered_packet);
  }

  // New packets received from the network.
  ForwardErrorCorrection::ReceivedPacket received_packet;
  received_packet.pkt = rtc::scoped_refptr<ForwardErrorCorrection::Packet>(
      new ForwardErrorCorrection::Packet());
  received_packet.pkt->data.SetSize(kPacketSize);
  received_packet.pkt->data.EnsureCapacity(IP_PACKET_SIZE);
  uint8_t* packet_buffer = received_packet.pkt->data.MutableData();
  uint8_t reordering;
  uint16_t seq_num_diff;
  uint8_t packet_type;
  uint8_t packet_loss;
  while (true) {
    if (!fuzz_buffer.ReadBytes(
            rtc::ArrayView<uint8_t>(packet_buffer, kPacketSize))) {
      return;
    }
    if (!fuzz_buffer.ReadUInt8(&reordering))
      return;
    if (!fuzz_buffer.ReadUInt16(&seq_num_diff))
      return;
    if (!fuzz_buffer.ReadUInt8(&packet_type))
      return;
    if (!fuzz_buffer.ReadUInt8(&packet_loss))
      return;

    if (reordering % 10 != 0)
      seq_num_diff = 0;

    if (packet_type % 2 == 0) {
      received_packet.is_fec = true;
      received_packet.ssrc = kFecSsrc;
      received_packet.seq_num = seq_num_diff + fec_seqnum++;

      // Overwrite parts of the FlexFEC header for fuzzing efficiency.
      packet_buffer[0] = 0;                                       // R, F bits.
      ByteWriter<uint8_t>::WriteBigEndian(&packet_buffer[8], 1);  // SSRCCount.
      ByteWriter<uint32_t>::WriteBigEndian(&packet_buffer[12],
                                           kMediaSsrc);  // SSRC_i.
      ByteWriter<uint16_t>::WriteBigEndian(
          &packet_buffer[16], original_media_seqnum);  // SN base_i.
    } else {
      received_packet.is_fec = false;
      received_packet.ssrc = kMediaSsrc;
      received_packet.seq_num = seq_num_diff + media_seqnum++;
    }

    if (packet_loss % 10 == 0)
      continue;

    fec->DecodeFec(received_packet, &recovered_packets);
  }
}

}  // namespace webrtc
