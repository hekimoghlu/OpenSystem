/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#include "modules/rtp_rtcp/source/rtp_util.h"

#include "test/gmock.h"

namespace webrtc {
namespace {

TEST(RtpUtilTest, IsRtpPacket) {
  constexpr uint8_t kMinimalisticRtpPacket[] = {0x80, 97, 0, 0,  //
                                                0,    0,  0, 0,  //
                                                0,    0,  0, 0};
  EXPECT_TRUE(IsRtpPacket(kMinimalisticRtpPacket));

  constexpr uint8_t kWrongRtpVersion[] = {0xc0, 97, 0, 0,  //
                                          0,    0,  0, 0,  //
                                          0,    0,  0, 0};
  EXPECT_FALSE(IsRtpPacket(kWrongRtpVersion));

  constexpr uint8_t kPacketWithPayloadForRtcp[] = {0x80, 200, 0, 0,  //
                                                   0,    0,   0, 0,  //
                                                   0,    0,   0, 0};
  EXPECT_FALSE(IsRtpPacket(kPacketWithPayloadForRtcp));

  constexpr uint8_t kTooSmallRtpPacket[] = {0x80, 97, 0, 0,  //
                                            0,    0,  0, 0,  //
                                            0,    0,  0};
  EXPECT_FALSE(IsRtpPacket(kTooSmallRtpPacket));

  EXPECT_FALSE(IsRtpPacket({}));
}

TEST(RtpUtilTest, IsRtcpPacket) {
  constexpr uint8_t kMinimalisticRtcpPacket[] = {0x80, 202, 0, 0};
  EXPECT_TRUE(IsRtcpPacket(kMinimalisticRtcpPacket));

  constexpr uint8_t kWrongRtpVersion[] = {0xc0, 202, 0, 0};
  EXPECT_FALSE(IsRtcpPacket(kWrongRtpVersion));

  constexpr uint8_t kPacketWithPayloadForRtp[] = {0x80, 225, 0, 0};
  EXPECT_FALSE(IsRtcpPacket(kPacketWithPayloadForRtp));

  constexpr uint8_t kTooSmallRtcpPacket[] = {0x80, 202, 0};
  EXPECT_FALSE(IsRtcpPacket(kTooSmallRtcpPacket));

  EXPECT_FALSE(IsRtcpPacket({}));
}

TEST(RtpUtilTest, ParseRtpPayloadType) {
  constexpr uint8_t kMinimalisticRtpPacket[] = {0x80, 97,   0,    0,  //
                                                0,    0,    0,    0,  //
                                                0x12, 0x34, 0x56, 0x78};
  EXPECT_EQ(ParseRtpPayloadType(kMinimalisticRtpPacket), 97);

  constexpr uint8_t kMinimalisticRtpPacketWithMarker[] = {
      0x80, 0x80 | 97, 0,    0,  //
      0,    0,         0,    0,  //
      0x12, 0x34,      0x56, 0x78};
  EXPECT_EQ(ParseRtpPayloadType(kMinimalisticRtpPacketWithMarker), 97);
}

TEST(RtpUtilTest, ParseRtpSequenceNumber) {
  constexpr uint8_t kMinimalisticRtpPacket[] = {0x80, 97, 0x12, 0x34,  //
                                                0,    0,  0,    0,     //
                                                0,    0,  0,    0};
  EXPECT_EQ(ParseRtpSequenceNumber(kMinimalisticRtpPacket), 0x1234);
}

TEST(RtpUtilTest, ParseRtpSsrc) {
  constexpr uint8_t kMinimalisticRtpPacket[] = {0x80, 97,   0,    0,  //
                                                0,    0,    0,    0,  //
                                                0x12, 0x34, 0x56, 0x78};
  EXPECT_EQ(ParseRtpSsrc(kMinimalisticRtpPacket), 0x12345678u);
}

}  // namespace
}  // namespace webrtc
