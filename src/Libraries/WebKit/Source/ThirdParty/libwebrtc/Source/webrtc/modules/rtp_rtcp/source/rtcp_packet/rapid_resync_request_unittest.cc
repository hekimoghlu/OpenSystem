/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#include "modules/rtp_rtcp/source/rtcp_packet/rapid_resync_request.h"

#include "test/gmock.h"
#include "test/gtest.h"
#include "test/rtcp_packet_parser.h"

using ::testing::ElementsAreArray;
using ::testing::make_tuple;
using webrtc::rtcp::RapidResyncRequest;

namespace webrtc {
namespace {
const uint32_t kSenderSsrc = 0x12345678;
const uint32_t kRemoteSsrc = 0x23456789;
// Manually created packet matching constants above.
const uint8_t kPacket[] = {0x85, 205,  0x00, 0x02, 0x12, 0x34,
                           0x56, 0x78, 0x23, 0x45, 0x67, 0x89};
}  // namespace

TEST(RtcpPacketRapidResyncRequestTest, Parse) {
  RapidResyncRequest mutable_parsed;
  EXPECT_TRUE(test::ParseSinglePacket(kPacket, &mutable_parsed));
  const RapidResyncRequest& parsed = mutable_parsed;

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  EXPECT_EQ(kRemoteSsrc, parsed.media_ssrc());
}

TEST(RtcpPacketRapidResyncRequestTest, Create) {
  RapidResyncRequest rrr;
  rrr.SetSenderSsrc(kSenderSsrc);
  rrr.SetMediaSsrc(kRemoteSsrc);

  rtc::Buffer packet = rrr.Build();

  EXPECT_THAT(make_tuple(packet.data(), packet.size()),
              ElementsAreArray(kPacket));
}

TEST(RtcpPacketRapidResyncRequestTest, ParseFailsOnTooSmallPacket) {
  const uint8_t kTooSmallPacket[] = {0x85, 205,  0x00, 0x01,
                                     0x12, 0x34, 0x56, 0x78};
  RapidResyncRequest parsed;
  EXPECT_FALSE(test::ParseSinglePacket(kTooSmallPacket, &parsed));
}

TEST(RtcpPacketRapidResyncRequestTest, ParseFailsOnTooLargePacket) {
  const uint8_t kTooLargePacket[] = {0x85, 205,  0x00, 0x03, 0x12, 0x34,
                                     0x56, 0x78, 0x32, 0x21, 0x65, 0x87,
                                     0x23, 0x45, 0x67, 0x89};
  RapidResyncRequest parsed;
  EXPECT_FALSE(test::ParseSinglePacket(kTooLargePacket, &parsed));
}
}  // namespace webrtc
