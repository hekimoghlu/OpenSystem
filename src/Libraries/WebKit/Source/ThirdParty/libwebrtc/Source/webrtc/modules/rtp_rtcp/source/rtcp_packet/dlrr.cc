/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#include "modules/rtp_rtcp/source/rtcp_packet/dlrr.h"

#include "modules/rtp_rtcp/source/byte_io.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {
namespace rtcp {
// DLRR Report Block (RFC 3611).
//
//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |     BT=5      |   reserved    |         block length          |
//  +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//  |                 SSRC_1 (SSRC of first receiver)               | sub-
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ block
//  |                         last RR (LRR)                         |   1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                   delay since last RR (DLRR)                  |
//  +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//  |                 SSRC_2 (SSRC of second receiver)              | sub-
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ block
//  :                               ...                             :   2

Dlrr::Dlrr() = default;

Dlrr::Dlrr(const Dlrr& other) = default;

Dlrr::~Dlrr() = default;

bool Dlrr::Parse(const uint8_t* buffer, uint16_t block_length_32bits) {
  RTC_DCHECK(buffer[0] == kBlockType);
  // kReserved = buffer[1];
  RTC_DCHECK_EQ(block_length_32bits,
                ByteReader<uint16_t>::ReadBigEndian(&buffer[2]));
  if (block_length_32bits % 3 != 0) {
    RTC_LOG(LS_WARNING) << "Invalid size for dlrr block.";
    return false;
  }

  size_t blocks_count = block_length_32bits / 3;
  const uint8_t* read_at = buffer + kBlockHeaderLength;
  sub_blocks_.resize(blocks_count);
  for (ReceiveTimeInfo& sub_block : sub_blocks_) {
    sub_block.ssrc = ByteReader<uint32_t>::ReadBigEndian(&read_at[0]);
    sub_block.last_rr = ByteReader<uint32_t>::ReadBigEndian(&read_at[4]);
    sub_block.delay_since_last_rr =
        ByteReader<uint32_t>::ReadBigEndian(&read_at[8]);
    read_at += kSubBlockLength;
  }
  return true;
}

size_t Dlrr::BlockLength() const {
  if (sub_blocks_.empty())
    return 0;
  return kBlockHeaderLength + kSubBlockLength * sub_blocks_.size();
}

void Dlrr::Create(uint8_t* buffer) const {
  if (sub_blocks_.empty())  // No subblocks, no need to write header either.
    return;
  // Create block header.
  const uint8_t kReserved = 0;
  buffer[0] = kBlockType;
  buffer[1] = kReserved;
  ByteWriter<uint16_t>::WriteBigEndian(
      &buffer[2], rtc::dchecked_cast<uint16_t>(3 * sub_blocks_.size()));
  // Create sub blocks.
  uint8_t* write_at = buffer + kBlockHeaderLength;
  for (const ReceiveTimeInfo& sub_block : sub_blocks_) {
    ByteWriter<uint32_t>::WriteBigEndian(&write_at[0], sub_block.ssrc);
    ByteWriter<uint32_t>::WriteBigEndian(&write_at[4], sub_block.last_rr);
    ByteWriter<uint32_t>::WriteBigEndian(&write_at[8],
                                         sub_block.delay_since_last_rr);
    write_at += kSubBlockLength;
  }
  RTC_DCHECK_EQ(buffer + BlockLength(), write_at);
}

}  // namespace rtcp
}  // namespace webrtc
