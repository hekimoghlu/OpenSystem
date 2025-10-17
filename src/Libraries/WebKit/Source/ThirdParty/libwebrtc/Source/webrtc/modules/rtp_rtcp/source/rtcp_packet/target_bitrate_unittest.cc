/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#include "modules/rtp_rtcp/source/rtcp_packet/target_bitrate.h"

#include "modules/rtp_rtcp/source/byte_io.h"
#include "modules/rtp_rtcp/source/rtcp_packet/extended_reports.h"
#include "rtc_base/buffer.h"
#include "test/gtest.h"
#include "test/rtcp_packet_parser.h"

namespace webrtc {
namespace {
using BitrateItem = rtcp::TargetBitrate::BitrateItem;
using rtcp::TargetBitrate;
using test::ParseSinglePacket;

constexpr uint32_t kSsrc = 0x12345678;

// clang-format off
const uint8_t kPacket[] = { TargetBitrate::kBlockType,  // Block ID.
                                  0x00,                 // Reserved.
                                        0x00, 0x04,     // Length = 4 words.
                            0x00, 0x01, 0x02, 0x03,     // S0T0 0x010203 kbps.
                            0x01, 0x02, 0x03, 0x04,     // S0T1 0x020304 kbps.
                            0x10, 0x03, 0x04, 0x05,     // S1T0 0x030405 kbps.
                            0x11, 0x04, 0x05, 0x06 };   // S1T1 0x040506 kbps.
constexpr size_t kPacketLengthBlocks = ((sizeof(kPacket) + 3) / 4) - 1;
// clang-format on

void ExpectBirateItemEquals(const BitrateItem& expected,
                            const BitrateItem& actual) {
  EXPECT_EQ(expected.spatial_layer, actual.spatial_layer);
  EXPECT_EQ(expected.temporal_layer, actual.temporal_layer);
  EXPECT_EQ(expected.target_bitrate_kbps, actual.target_bitrate_kbps);
}

void CheckBitrateItems(const std::vector<BitrateItem>& bitrates) {
  EXPECT_EQ(4U, bitrates.size());
  ExpectBirateItemEquals(BitrateItem(0, 0, 0x010203), bitrates[0]);
  ExpectBirateItemEquals(BitrateItem(0, 1, 0x020304), bitrates[1]);
  ExpectBirateItemEquals(BitrateItem(1, 0, 0x030405), bitrates[2]);
  ExpectBirateItemEquals(BitrateItem(1, 1, 0x040506), bitrates[3]);
}

}  // namespace

TEST(TargetBitrateTest, Parse) {
  TargetBitrate target_bitrate;
  target_bitrate.Parse(kPacket, kPacketLengthBlocks);
  CheckBitrateItems(target_bitrate.GetTargetBitrates());
}

TEST(TargetBitrateTest, FullPacket) {
  const size_t kXRHeaderSize = 8;  // RTCP header (4) + SSRC (4).
  const size_t kTotalSize = kXRHeaderSize + sizeof(kPacket);
  uint8_t kRtcpPacket[kTotalSize] = {2 << 6, 207,  0x00, (kTotalSize / 4) - 1,
                                     0x12,   0x34, 0x56, 0x78};  // SSRC.
  memcpy(&kRtcpPacket[kXRHeaderSize], kPacket, sizeof(kPacket));
  rtcp::ExtendedReports xr;
  EXPECT_TRUE(ParseSinglePacket(kRtcpPacket, &xr));
  EXPECT_EQ(kSsrc, xr.sender_ssrc());
  const std::optional<TargetBitrate>& target_bitrate = xr.target_bitrate();
  ASSERT_TRUE(static_cast<bool>(target_bitrate));
  CheckBitrateItems(target_bitrate->GetTargetBitrates());
}

TEST(TargetBitrateTest, Create) {
  TargetBitrate target_bitrate;
  target_bitrate.AddTargetBitrate(0, 0, 0x010203);
  target_bitrate.AddTargetBitrate(0, 1, 0x020304);
  target_bitrate.AddTargetBitrate(1, 0, 0x030405);
  target_bitrate.AddTargetBitrate(1, 1, 0x040506);

  uint8_t buffer[sizeof(kPacket)] = {};
  ASSERT_EQ(sizeof(kPacket), target_bitrate.BlockLength());
  target_bitrate.Create(buffer);

  EXPECT_EQ(0, memcmp(kPacket, buffer, sizeof(kPacket)));
}

TEST(TargetBitrateTest, ParseNullBitratePacket) {
  const uint8_t kNullPacket[] = {TargetBitrate::kBlockType, 0x00, 0x00, 0x00};
  TargetBitrate target_bitrate;
  target_bitrate.Parse(kNullPacket, 0);
  EXPECT_TRUE(target_bitrate.GetTargetBitrates().empty());
}

}  // namespace webrtc
