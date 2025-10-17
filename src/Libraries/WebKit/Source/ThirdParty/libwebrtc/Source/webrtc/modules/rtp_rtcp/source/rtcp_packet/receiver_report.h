/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RECEIVER_REPORT_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RECEIVER_REPORT_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "modules/rtp_rtcp/source/rtcp_packet.h"
#include "modules/rtp_rtcp/source/rtcp_packet/report_block.h"

namespace webrtc {
namespace rtcp {
class CommonHeader;

class ReceiverReport : public RtcpPacket {
 public:
  static constexpr uint8_t kPacketType = 201;
  static constexpr size_t kMaxNumberOfReportBlocks = 0x1f;

  ReceiverReport();
  ReceiverReport(const ReceiverReport&);
  ~ReceiverReport() override;

  // Parse assumes header is already parsed and validated.
  bool Parse(const CommonHeader& packet);

  bool AddReportBlock(const ReportBlock& block);
  bool SetReportBlocks(std::vector<ReportBlock> blocks);

  const std::vector<ReportBlock>& report_blocks() const {
    return report_blocks_;
  }

  size_t BlockLength() const override;

  bool Create(uint8_t* packet,
              size_t* index,
              size_t max_length,
              PacketReadyCallback callback) const override;

 private:
  static constexpr size_t kRrBaseLength = 4;

  std::vector<ReportBlock> report_blocks_;
};

}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_RECEIVER_REPORT_H_
