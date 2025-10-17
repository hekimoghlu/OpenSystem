/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#include "modules/rtp_rtcp/source/fec_private_tables_bursty.h"

#include "modules/rtp_rtcp/source/fec_private_tables_random.h"
#include "modules/rtp_rtcp/source/forward_error_correction_internal.h"
#include "test/gtest.h"

namespace webrtc {
namespace {
constexpr uint8_t kMaskRandom15_6[] = {0x82, 0x08, 0x41, 0x04, 0x20, 0x82,
                                       0x10, 0x40, 0x08, 0x20, 0x04, 0x10};
}

namespace fec_private_tables {

using internal::LookUpInFecTable;

TEST(FecTable, TestBurstyLookup) {
  rtc::ArrayView<const uint8_t> result;
  result = LookUpInFecTable(&kPacketMaskBurstyTbl[0], 0, 0);
  // Should match kMaskBursty1_1.
  EXPECT_EQ(2u, result.size());
  EXPECT_EQ(0x80u, result[0]);

  result = LookUpInFecTable(&kPacketMaskBurstyTbl[0], 3, 0);
  // Should match kMaskBursty4_1.
  EXPECT_EQ(2u, result.size());
  EXPECT_EQ(0xf0u, result[0]);
  EXPECT_EQ(0x00u, result[1]);

  result = LookUpInFecTable(&kPacketMaskBurstyTbl[0], 1, 1);
  // Should match kMaskBursty2_2.
  EXPECT_EQ(4u, result.size());
  EXPECT_EQ(0x80u, result[0]);
  EXPECT_EQ(0xc0u, result[2]);

  result = LookUpInFecTable(&kPacketMaskBurstyTbl[0], 11, 11);
  // Should match kMaskBursty12_12.
  EXPECT_EQ(24u, result.size());
  EXPECT_EQ(0x80u, result[0]);
  EXPECT_EQ(0x30u, result[23]);
}

TEST(FecTable, TestRandomLookup) {
  rtc::ArrayView<const uint8_t> result;
  result = LookUpInFecTable(&kPacketMaskRandomTbl[0], 0, 0);
  EXPECT_EQ(2u, result.size());
  EXPECT_EQ(0x80u, result[0]);
  EXPECT_EQ(0x00u, result[1]);

  result = LookUpInFecTable(&kPacketMaskRandomTbl[0], 4, 1);
  // kMaskRandom5_2.
  EXPECT_EQ(4u, result.size());
  EXPECT_EQ(0xa8u, result[0]);
  EXPECT_EQ(0xd0u, result[2]);
}

TEST(FecTable, TestRandomGenerated) {
  FecMaskType fec_mask_type = kFecMaskRandom;
  int num_media_packets = 15;
  int num_fec_packets = 6;
  size_t mask_size = sizeof(kMaskRandom15_6) / sizeof(uint8_t);
  internal::PacketMaskTable mask_table(fec_mask_type, num_media_packets);
  rtc::ArrayView<const uint8_t> mask =
      mask_table.LookUp(num_media_packets, num_fec_packets);
  EXPECT_EQ(mask.size(), mask_size);
  for (size_t i = 0; i < mask_size; ++i) {
    EXPECT_EQ(mask[i], kMaskRandom15_6[i]);
  }
}

}  // namespace fec_private_tables
}  // namespace webrtc
