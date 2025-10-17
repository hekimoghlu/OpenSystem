/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
#include "modules/rtp_rtcp/source/rtcp_packet/bye.h"

#include "test/gmock.h"
#include "test/gtest.h"
#include "test/rtcp_packet_parser.h"

using ::testing::ElementsAre;
using webrtc::rtcp::Bye;

namespace webrtc {
namespace {
const uint32_t kSenderSsrc = 0x12345678;
const uint32_t kCsrc1 = 0x22232425;
const uint32_t kCsrc2 = 0x33343536;
}  // namespace

TEST(RtcpPacketByeTest, CreateAndParseWithoutReason) {
  Bye bye;
  bye.SetSenderSsrc(kSenderSsrc);

  rtc::Buffer raw = bye.Build();
  Bye parsed_bye;
  EXPECT_TRUE(test::ParseSinglePacket(raw, &parsed_bye));

  EXPECT_EQ(kSenderSsrc, parsed_bye.sender_ssrc());
  EXPECT_TRUE(parsed_bye.csrcs().empty());
  EXPECT_TRUE(parsed_bye.reason().empty());
}

TEST(RtcpPacketByeTest, CreateAndParseWithCsrcs) {
  Bye bye;
  bye.SetSenderSsrc(kSenderSsrc);
  EXPECT_TRUE(bye.SetCsrcs({kCsrc1, kCsrc2}));
  EXPECT_TRUE(bye.reason().empty());

  rtc::Buffer raw = bye.Build();
  Bye parsed_bye;
  EXPECT_TRUE(test::ParseSinglePacket(raw, &parsed_bye));

  EXPECT_EQ(kSenderSsrc, parsed_bye.sender_ssrc());
  EXPECT_THAT(parsed_bye.csrcs(), ElementsAre(kCsrc1, kCsrc2));
  EXPECT_TRUE(parsed_bye.reason().empty());
}

TEST(RtcpPacketByeTest, CreateAndParseWithCsrcsAndAReason) {
  Bye bye;
  const std::string kReason = "Some Reason";

  bye.SetSenderSsrc(kSenderSsrc);
  EXPECT_TRUE(bye.SetCsrcs({kCsrc1, kCsrc2}));
  bye.SetReason(kReason);

  rtc::Buffer raw = bye.Build();
  Bye parsed_bye;
  EXPECT_TRUE(test::ParseSinglePacket(raw, &parsed_bye));

  EXPECT_EQ(kSenderSsrc, parsed_bye.sender_ssrc());
  EXPECT_THAT(parsed_bye.csrcs(), ElementsAre(kCsrc1, kCsrc2));
  EXPECT_EQ(kReason, parsed_bye.reason());
}

TEST(RtcpPacketByeTest, CreateWithTooManyCsrcs) {
  Bye bye;
  bye.SetSenderSsrc(kSenderSsrc);
  const int kMaxCsrcs = (1 << 5) - 2;  // 5 bit len, first item is sender SSRC.
  EXPECT_TRUE(bye.SetCsrcs(std::vector<uint32_t>(kMaxCsrcs, kCsrc1)));
  EXPECT_FALSE(bye.SetCsrcs(std::vector<uint32_t>(kMaxCsrcs + 1, kCsrc1)));
}

TEST(RtcpPacketByeTest, CreateAndParseWithAReason) {
  Bye bye;
  const std::string kReason = "Some Random Reason";

  bye.SetSenderSsrc(kSenderSsrc);
  bye.SetReason(kReason);

  rtc::Buffer raw = bye.Build();
  Bye parsed_bye;
  EXPECT_TRUE(test::ParseSinglePacket(raw, &parsed_bye));

  EXPECT_EQ(kSenderSsrc, parsed_bye.sender_ssrc());
  EXPECT_TRUE(parsed_bye.csrcs().empty());
  EXPECT_EQ(kReason, parsed_bye.reason());
}

TEST(RtcpPacketByeTest, CreateAndParseWithReasons) {
  // Test that packet creation/parsing behave with reasons of different length
  // both when it require padding and when it does not.
  for (size_t reminder = 0; reminder < 4; ++reminder) {
    const std::string kReason(4 + reminder, 'a' + reminder);
    Bye bye;
    bye.SetSenderSsrc(kSenderSsrc);
    bye.SetReason(kReason);

    rtc::Buffer raw = bye.Build();
    Bye parsed_bye;
    EXPECT_TRUE(test::ParseSinglePacket(raw, &parsed_bye));

    EXPECT_EQ(kReason, parsed_bye.reason());
  }
}

TEST(RtcpPacketByeTest, ParseEmptyPacket) {
  uint8_t kEmptyPacket[] = {0x80, Bye::kPacketType, 0, 0};
  Bye parsed_bye;
  EXPECT_TRUE(test::ParseSinglePacket(kEmptyPacket, &parsed_bye));
  EXPECT_EQ(0u, parsed_bye.sender_ssrc());
  EXPECT_TRUE(parsed_bye.csrcs().empty());
  EXPECT_TRUE(parsed_bye.reason().empty());
}

TEST(RtcpPacketByeTest, ParseFailOnInvalidSrcCount) {
  Bye bye;
  bye.SetSenderSsrc(kSenderSsrc);

  rtc::Buffer raw = bye.Build();
  raw[0]++;  // Damage the packet: increase ssrc count by one.

  Bye parsed_bye;
  EXPECT_FALSE(test::ParseSinglePacket(raw, &parsed_bye));
}

TEST(RtcpPacketByeTest, ParseFailOnInvalidReasonLength) {
  Bye bye;
  bye.SetSenderSsrc(kSenderSsrc);
  bye.SetReason("18 characters long");

  rtc::Buffer raw = bye.Build();
  // Damage the packet: decrease payload size by 4 bytes
  raw[3]--;
  raw.SetSize(raw.size() - 4);

  Bye parsed_bye;
  EXPECT_FALSE(test::ParseSinglePacket(raw, &parsed_bye));
}

}  // namespace webrtc
