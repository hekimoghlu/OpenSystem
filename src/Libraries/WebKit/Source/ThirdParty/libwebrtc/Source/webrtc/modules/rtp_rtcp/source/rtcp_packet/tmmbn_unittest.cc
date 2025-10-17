/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#include "modules/rtp_rtcp/source/rtcp_packet/tmmbn.h"

#include "test/gmock.h"
#include "test/gtest.h"
#include "test/rtcp_packet_parser.h"

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::make_tuple;
using webrtc::rtcp::TmmbItem;
using webrtc::rtcp::Tmmbn;

namespace webrtc {
namespace {
const uint32_t kSenderSsrc = 0x12345678;
const uint32_t kRemoteSsrc = 0x23456789;
const uint32_t kBitrateBps = 312000;
const uint16_t kOverhead = 0x1fe;
const uint8_t kPacket[] = {0x84, 205,  0x00, 0x04, 0x12, 0x34, 0x56,
                           0x78, 0x00, 0x00, 0x00, 0x00, 0x23, 0x45,
                           0x67, 0x89, 0x0a, 0x61, 0x61, 0xfe};
}  // namespace

TEST(RtcpPacketTmmbnTest, Create) {
  Tmmbn tmmbn;
  tmmbn.SetSenderSsrc(kSenderSsrc);
  tmmbn.AddTmmbr(TmmbItem(kRemoteSsrc, kBitrateBps, kOverhead));

  rtc::Buffer packet = tmmbn.Build();

  EXPECT_THAT(make_tuple(packet.data(), packet.size()),
              ElementsAreArray(kPacket));
}

TEST(RtcpPacketTmmbnTest, Parse) {
  Tmmbn tmmbn;
  EXPECT_TRUE(test::ParseSinglePacket(kPacket, &tmmbn));

  const Tmmbn& parsed = tmmbn;

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  ASSERT_EQ(1u, parsed.items().size());
  EXPECT_EQ(kRemoteSsrc, parsed.items().front().ssrc());
  EXPECT_EQ(kBitrateBps, parsed.items().front().bitrate_bps());
  EXPECT_EQ(kOverhead, parsed.items().front().packet_overhead());
}

TEST(RtcpPacketTmmbnTest, CreateAndParseWithoutItems) {
  Tmmbn tmmbn;
  tmmbn.SetSenderSsrc(kSenderSsrc);

  rtc::Buffer packet = tmmbn.Build();
  Tmmbn parsed;
  EXPECT_TRUE(test::ParseSinglePacket(packet, &parsed));

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  EXPECT_THAT(parsed.items(), IsEmpty());
}

TEST(RtcpPacketTmmbnTest, CreateAndParseWithTwoItems) {
  Tmmbn tmmbn;
  tmmbn.SetSenderSsrc(kSenderSsrc);
  tmmbn.AddTmmbr(TmmbItem(kRemoteSsrc, kBitrateBps, kOverhead));
  tmmbn.AddTmmbr(TmmbItem(kRemoteSsrc + 1, 4 * kBitrateBps, 40));

  rtc::Buffer packet = tmmbn.Build();
  Tmmbn parsed;
  EXPECT_TRUE(test::ParseSinglePacket(packet, &parsed));

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  EXPECT_EQ(2u, parsed.items().size());
  EXPECT_EQ(kRemoteSsrc, parsed.items()[0].ssrc());
  EXPECT_EQ(kBitrateBps, parsed.items()[0].bitrate_bps());
  EXPECT_EQ(kOverhead, parsed.items()[0].packet_overhead());
  EXPECT_EQ(kRemoteSsrc + 1, parsed.items()[1].ssrc());
  EXPECT_EQ(4 * kBitrateBps, parsed.items()[1].bitrate_bps());
  EXPECT_EQ(40U, parsed.items()[1].packet_overhead());
}

TEST(RtcpPacketTmmbnTest, ParseFailsOnTooSmallPacket) {
  const uint8_t kSmallPacket[] = {0x84, 205,  0x00, 0x01,
                                  0x12, 0x34, 0x56, 0x78};
  Tmmbn tmmbn;
  EXPECT_FALSE(test::ParseSinglePacket(kSmallPacket, &tmmbn));
}

TEST(RtcpPacketTmmbnTest, ParseFailsOnUnAlignedPacket) {
  const uint8_t kUnalignedPacket[] = {0x84, 205,  0x00, 0x03, 0x12, 0x34,
                                      0x56, 0x78, 0x00, 0x00, 0x00, 0x00,
                                      0x23, 0x45, 0x67, 0x89};

  Tmmbn tmmbn;
  EXPECT_FALSE(test::ParseSinglePacket(kUnalignedPacket, &tmmbn));
}
}  // namespace webrtc
