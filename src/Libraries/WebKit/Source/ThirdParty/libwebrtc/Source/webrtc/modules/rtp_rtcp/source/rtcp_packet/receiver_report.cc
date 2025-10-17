/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#include "modules/rtp_rtcp/source/rtcp_packet/receiver_report.h"

#include <utility>

#include "modules/rtp_rtcp/source/byte_io.h"
#include "modules/rtp_rtcp/source/rtcp_packet/common_header.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace rtcp {
// RTCP receiver report (RFC 3550).
//
//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |V=2|P|    RC   |   PT=RR=201   |             length            |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                     SSRC of packet sender                     |
//  +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//  |                         report block(s)                       |
//  |                            ....                               |

ReceiverReport::ReceiverReport() = default;

ReceiverReport::ReceiverReport(const ReceiverReport& rhs) = default;

ReceiverReport::~ReceiverReport() = default;

bool ReceiverReport::Parse(const CommonHeader& packet) {
  RTC_DCHECK_EQ(packet.type(), kPacketType);

  const uint8_t report_blocks_count = packet.count();

  if (packet.payload_size_bytes() <
      kRrBaseLength + report_blocks_count * ReportBlock::kLength) {
    RTC_LOG(LS_WARNING) << "Packet is too small to contain all the data.";
    return false;
  }

  SetSenderSsrc(ByteReader<uint32_t>::ReadBigEndian(packet.payload()));

  const uint8_t* next_report_block = packet.payload() + kRrBaseLength;

  report_blocks_.resize(report_blocks_count);
  for (ReportBlock& block : report_blocks_) {
    block.Parse(next_report_block, ReportBlock::kLength);
    next_report_block += ReportBlock::kLength;
  }

  RTC_DCHECK_LE(next_report_block - packet.payload(),
                static_cast<ptrdiff_t>(packet.payload_size_bytes()));
  return true;
}

size_t ReceiverReport::BlockLength() const {
  return kHeaderLength + kRrBaseLength +
         report_blocks_.size() * ReportBlock::kLength;
}

bool ReceiverReport::Create(uint8_t* packet,
                            size_t* index,
                            size_t max_length,
                            PacketReadyCallback callback) const {
  while (*index + BlockLength() > max_length) {
    if (!OnBufferFull(packet, index, callback))
      return false;
  }
  CreateHeader(report_blocks_.size(), kPacketType, HeaderLength(), packet,
               index);
  ByteWriter<uint32_t>::WriteBigEndian(packet + *index, sender_ssrc());
  *index += kRrBaseLength;
  for (const ReportBlock& block : report_blocks_) {
    block.Create(packet + *index);
    *index += ReportBlock::kLength;
  }
  return true;
}

bool ReceiverReport::AddReportBlock(const ReportBlock& block) {
  if (report_blocks_.size() >= kMaxNumberOfReportBlocks) {
    RTC_LOG(LS_WARNING) << "Max report blocks reached.";
    return false;
  }
  report_blocks_.push_back(block);
  return true;
}

bool ReceiverReport::SetReportBlocks(std::vector<ReportBlock> blocks) {
  if (blocks.size() > kMaxNumberOfReportBlocks) {
    RTC_LOG(LS_WARNING) << "Too many report blocks (" << blocks.size()
                        << ") for receiver report.";
    return false;
  }
  report_blocks_ = std::move(blocks);
  return true;
}

}  // namespace rtcp
}  // namespace webrtc
