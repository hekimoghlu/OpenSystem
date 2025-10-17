/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
#include "modules/rtp_rtcp/source/rtcp_packet/rrtr.h"

#include "modules/rtp_rtcp/source/byte_io.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace rtcp {
// Receiver Reference Time Report Block (RFC 3611).
//
//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |     BT=4      |   reserved    |       block length = 2        |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |              NTP timestamp, most significant word             |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |             NTP timestamp, least significant word             |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

void Rrtr::Parse(const uint8_t* buffer) {
  RTC_DCHECK(buffer[0] == kBlockType);
  // reserved = buffer[1];
  RTC_DCHECK(ByteReader<uint16_t>::ReadBigEndian(&buffer[2]) == kBlockLength);
  uint32_t seconds = ByteReader<uint32_t>::ReadBigEndian(&buffer[4]);
  uint32_t fraction = ByteReader<uint32_t>::ReadBigEndian(&buffer[8]);
  ntp_.Set(seconds, fraction);
}

void Rrtr::Create(uint8_t* buffer) const {
  const uint8_t kReserved = 0;
  buffer[0] = kBlockType;
  buffer[1] = kReserved;
  ByteWriter<uint16_t>::WriteBigEndian(&buffer[2], kBlockLength);
  ByteWriter<uint32_t>::WriteBigEndian(&buffer[4], ntp_.seconds());
  ByteWriter<uint32_t>::WriteBigEndian(&buffer[8], ntp_.fractions());
}

}  // namespace rtcp
}  // namespace webrtc
