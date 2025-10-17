/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include "modules/rtp_rtcp/source/rtcp_packet/common_header.h"

#include "modules/rtp_rtcp/source/byte_io.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace rtcp {
//    0                   1           1       2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// 0 |V=2|P|   C/F   |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// 1                 |  Packet Type  |
//   ----------------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// 2                                 |             length            |
//   --------------------------------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
// Common header for all RTCP packets, 4 octets.
bool CommonHeader::Parse(const uint8_t* buffer, size_t size_bytes) {
  const uint8_t kVersion = 2;

  if (size_bytes < kHeaderSizeBytes) {
    RTC_LOG(LS_WARNING)
        << "Too little data (" << size_bytes << " byte"
        << (size_bytes != 1 ? "s" : "")
        << ") remaining in buffer to parse RTCP header (4 bytes).";
    return false;
  }

  uint8_t version = buffer[0] >> 6;
  if (version != kVersion) {
    RTC_LOG(LS_WARNING) << "Invalid RTCP header: Version must be "
                        << static_cast<int>(kVersion) << " but was "
                        << static_cast<int>(version);
    return false;
  }

  bool has_padding = (buffer[0] & 0x20) != 0;
  count_or_format_ = buffer[0] & 0x1F;
  packet_type_ = buffer[1];
  payload_size_ = ByteReader<uint16_t>::ReadBigEndian(&buffer[2]) * 4;
  payload_ = buffer + kHeaderSizeBytes;
  padding_size_ = 0;

  if (size_bytes < kHeaderSizeBytes + payload_size_) {
    RTC_LOG(LS_WARNING) << "Buffer too small (" << size_bytes
                        << " bytes) to fit an RtcpPacket with a header and "
                        << payload_size_ << " bytes.";
    return false;
  }

  if (has_padding) {
    if (payload_size_ == 0) {
      RTC_LOG(LS_WARNING)
          << "Invalid RTCP header: Padding bit set but 0 payload "
             "size specified.";
      return false;
    }

    padding_size_ = payload_[payload_size_ - 1];
    if (padding_size_ == 0) {
      RTC_LOG(LS_WARNING)
          << "Invalid RTCP header: Padding bit set but 0 padding "
             "size specified.";
      return false;
    }
    if (padding_size_ > payload_size_) {
      RTC_LOG(LS_WARNING) << "Invalid RTCP header: Too many padding bytes ("
                          << padding_size_ << ") for a packet payload size of "
                          << payload_size_ << " bytes.";
      return false;
    }
    payload_size_ -= padding_size_;
  }
  return true;
}
}  // namespace rtcp
}  // namespace webrtc
