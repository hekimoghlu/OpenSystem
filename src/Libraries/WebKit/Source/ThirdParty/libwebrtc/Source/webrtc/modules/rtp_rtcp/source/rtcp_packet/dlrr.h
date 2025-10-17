/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_DLRR_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_DLRR_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace webrtc {
namespace rtcp {
struct ReceiveTimeInfo {
  // RFC 3611 4.5
  ReceiveTimeInfo() : ssrc(0), last_rr(0), delay_since_last_rr(0) {}
  ReceiveTimeInfo(uint32_t ssrc, uint32_t last_rr, uint32_t delay)
      : ssrc(ssrc), last_rr(last_rr), delay_since_last_rr(delay) {}

  uint32_t ssrc;
  uint32_t last_rr;
  uint32_t delay_since_last_rr;
};

inline bool operator==(const ReceiveTimeInfo& lhs, const ReceiveTimeInfo& rhs) {
  return lhs.ssrc == rhs.ssrc && lhs.last_rr == rhs.last_rr &&
         lhs.delay_since_last_rr == rhs.delay_since_last_rr;
}

inline bool operator!=(const ReceiveTimeInfo& lhs, const ReceiveTimeInfo& rhs) {
  return !(lhs == rhs);
}

// DLRR Report Block: Delay since the Last Receiver Report (RFC 3611).
class Dlrr {
 public:
  static constexpr uint8_t kBlockType = 5;

  Dlrr();
  Dlrr(const Dlrr& other);
  ~Dlrr();

  Dlrr& operator=(const Dlrr& other) = default;

  // Dlrr without items treated same as no dlrr block.
  explicit operator bool() const { return !sub_blocks_.empty(); }

  // Second parameter is value read from block header,
  // i.e. size of block in 32bits excluding block header itself.
  bool Parse(const uint8_t* buffer, uint16_t block_length_32bits);

  size_t BlockLength() const;
  // Fills buffer with the Dlrr.
  // Consumes BlockLength() bytes.
  void Create(uint8_t* buffer) const;

  void ClearItems() { sub_blocks_.clear(); }
  void AddDlrrItem(const ReceiveTimeInfo& time_info) {
    sub_blocks_.push_back(time_info);
  }

  const std::vector<ReceiveTimeInfo>& sub_blocks() const { return sub_blocks_; }

 private:
  static constexpr size_t kBlockHeaderLength = 4;
  static constexpr size_t kSubBlockLength = 12;

  std::vector<ReceiveTimeInfo> sub_blocks_;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_DLRR_H_
