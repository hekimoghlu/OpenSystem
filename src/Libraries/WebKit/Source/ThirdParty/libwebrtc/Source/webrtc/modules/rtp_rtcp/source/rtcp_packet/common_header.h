/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_COMMON_HEADER_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_COMMON_HEADER_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {
namespace rtcp {
class CommonHeader {
 public:
  static constexpr size_t kHeaderSizeBytes = 4;

  CommonHeader() {}
  CommonHeader(const CommonHeader&) = default;
  CommonHeader& operator=(const CommonHeader&) = default;

  bool Parse(const uint8_t* buffer, size_t size_bytes);

  uint8_t type() const { return packet_type_; }
  // Depending on packet type same header field can be used either as count or
  // as feedback message type (fmt). Caller expected to know how it is used.
  uint8_t fmt() const { return count_or_format_; }
  uint8_t count() const { return count_or_format_; }
  size_t payload_size_bytes() const { return payload_size_; }
  const uint8_t* payload() const { return payload_; }
  size_t packet_size() const {
    return kHeaderSizeBytes + payload_size_ + padding_size_;
  }
  // Returns pointer to the next RTCP packet in compound packet.
  const uint8_t* NextPacket() const {
    return payload_ + payload_size_ + padding_size_;
  }

 private:
  uint8_t packet_type_ = 0;
  uint8_t count_or_format_ = 0;
  uint8_t padding_size_ = 0;
  uint32_t payload_size_ = 0;
  const uint8_t* payload_ = nullptr;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_COMMON_HEADER_H_
