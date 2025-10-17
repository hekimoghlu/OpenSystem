/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#include "modules/rtp_rtcp/source/rtcp_packet/tmmbr.h"

#include "test/gmock.h"
#include "test/gtest.h"
#include "test/rtcp_packet_parser.h"

using ::testing::ElementsAreArray;
using ::testing::make_tuple;
using webrtc::rtcp::TmmbItem;
using webrtc::rtcp::Tmmbr;

namespace webrtc {
namespace {
const uint32_t kSenderSsrc = 0x12345678;
const uint32_t kRemoteSsrc = 0x23456789;
const uint32_t kBitrateBps = 312000;
const uint16_t kOverhead = 0x1fe;
const uint8_t kPacket[] = {0x83, 205,  0x00, 0x04, 0x12, 0x34, 0x56,
                           0x78, 0x00, 0x00, 0x00, 0x00, 0x23, 0x45,
                           0x67, 0x89, 0x0a, 0x61, 0x61, 0xfe};
}  // namespace

TEST(RtcpPacketTmmbrTest, Create) {
  Tmmbr tmmbr;
  tmmbr.SetSenderSsrc(kSenderSsrc);
  tmmbr.AddTmmbr(TmmbItem(kRemoteSsrc, kBitrateBps, kOverhead));

  rtc::Buffer packet = tmmbr.Build();

  EXPECT_THAT(make_tuple(packet.data(), packet.size()),
              ElementsAreArray(kPacket));
}

TEST(RtcpPacketTmmbrTest, Parse) {
  Tmmbr tmmbr;
  EXPECT_TRUE(test::ParseSinglePacket(kPacket, &tmmbr));
  const Tmmbr& parsed = tmmbr;

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  ASSERT_EQ(1u, parsed.requests().size());
  EXPECT_EQ(kRemoteSsrc, parsed.requests().front().ssrc());
  EXPECT_EQ(kBitrateBps, parsed.requests().front().bitrate_bps());
  EXPECT_EQ(kOverhead, parsed.requests().front().packet_overhead());
}

TEST(RtcpPacketTmmbrTest, CreateAndParseWithTwoEntries) {
  Tmmbr tmmbr;
  tmmbr.SetSenderSsrc(kSenderSsrc);
  tmmbr.AddTmmbr(TmmbItem(kRemoteSsrc, kBitrateBps, kOverhead));
  tmmbr.AddTmmbr(TmmbItem(kRemoteSsrc + 1, 4 * kBitrateBps, kOverhead + 1));

  rtc::Buffer packet = tmmbr.Build();

  Tmmbr parsed;
  EXPECT_TRUE(test::ParseSinglePacket(packet, &parsed));

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  EXPECT_EQ(2u, parsed.requests().size());
  EXPECT_EQ(kRemoteSsrc, parsed.requests()[0].ssrc());
  EXPECT_EQ(kRemoteSsrc + 1, parsed.requests()[1].ssrc());
}

TEST(RtcpPacketTmmbrTest, ParseFailsWithoutItems) {
  const uint8_t kZeroItemsPacket[] = {0x83, 205,  0x00, 0x02, 0x12, 0x34,
                                      0x56, 0x78, 0x00, 0x00, 0x00, 0x00};

  Tmmbr tmmbr;
  EXPECT_FALSE(test::ParseSinglePacket(kZeroItemsPacket, &tmmbr));
}

TEST(RtcpPacketTmmbrTest, ParseFailsOnUnAlignedPacket) {
  const uint8_t kUnalignedPacket[] = {
      0x83, 205,  0x00, 0x05, 0x12, 0x34, 0x56, 0x78, 0x00, 0x00, 0x00, 0x00,
      0x23, 0x45, 0x67, 0x89, 0x0a, 0x61, 0x61, 0xfe, 0x34, 0x56, 0x78, 0x9a};

  Tmmbr tmmbr;
  EXPECT_FALSE(test::ParseSinglePacket(kUnalignedPacket, &tmmbr));
}
}  // namespace webrtc
