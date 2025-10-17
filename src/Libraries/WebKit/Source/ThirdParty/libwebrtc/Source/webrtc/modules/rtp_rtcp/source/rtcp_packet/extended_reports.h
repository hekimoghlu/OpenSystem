/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_EXTENDED_REPORTS_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_EXTENDED_REPORTS_H_

#include <optional>
#include <vector>

#include "modules/rtp_rtcp/source/rtcp_packet.h"
#include "modules/rtp_rtcp/source/rtcp_packet/dlrr.h"
#include "modules/rtp_rtcp/source/rtcp_packet/rrtr.h"
#include "modules/rtp_rtcp/source/rtcp_packet/target_bitrate.h"

namespace webrtc {
namespace rtcp {
class CommonHeader;

// From RFC 3611: RTP Control Protocol Extended Reports (RTCP XR).
class ExtendedReports : public RtcpPacket {
 public:
  static constexpr uint8_t kPacketType = 207;
  static constexpr size_t kMaxNumberOfDlrrItems = 50;

  ExtendedReports();
  ExtendedReports(const ExtendedReports& xr);
  ~ExtendedReports() override;

  // Parse assumes header is already parsed and validated.
  bool Parse(const CommonHeader& packet);

  void SetRrtr(const Rrtr& rrtr);
  bool AddDlrrItem(const ReceiveTimeInfo& time_info);
  void SetTargetBitrate(const TargetBitrate& target_bitrate);

  const std::optional<Rrtr>& rrtr() const { return rrtr_block_; }
  const Dlrr& dlrr() const { return dlrr_block_; }
  const std::optional<TargetBitrate>& target_bitrate() const {
    return target_bitrate_;
  }

  size_t BlockLength() const override;

  bool Create(uint8_t* packet,
              size_t* index,
              size_t max_length,
              PacketReadyCallback callback) const override;

 private:
  static constexpr size_t kXrBaseLength = 4;

  size_t RrtrLength() const { return rrtr_block_ ? Rrtr::kLength : 0; }
  size_t DlrrLength() const { return dlrr_block_.BlockLength(); }
  size_t TargetBitrateLength() const;

  void ParseRrtrBlock(const uint8_t* block, uint16_t block_length);
  void ParseDlrrBlock(const uint8_t* block, uint16_t block_length);
  void ParseTargetBitrateBlock(const uint8_t* block, uint16_t block_length);

  std::optional<Rrtr> rrtr_block_;
  Dlrr dlrr_block_;  // Dlrr without items treated same as no dlrr block.
  std::optional<TargetBitrate> target_bitrate_;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_EXTENDED_REPORTS_H_
