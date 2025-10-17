/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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

#include "test/gtest.h"

using webrtc::rtcp::CommonHeader;

namespace webrtc {

TEST(RtcpCommonHeaderTest, TooSmallBuffer) {
  uint8_t buffer[] = {0x80, 0x00, 0x00, 0x00};
  CommonHeader header;
  // Buffer needs to be able to hold the header.
  EXPECT_FALSE(header.Parse(buffer, 0));
  EXPECT_FALSE(header.Parse(buffer, 1));
  EXPECT_FALSE(header.Parse(buffer, 2));
  EXPECT_FALSE(header.Parse(buffer, 3));
  EXPECT_TRUE(header.Parse(buffer, 4));
}

TEST(RtcpCommonHeaderTest, Version) {
  uint8_t buffer[] = {0x00, 0x00, 0x00, 0x00};
  CommonHeader header;
  // Version 2 is the only allowed.
  buffer[0] = 0 << 6;
  EXPECT_FALSE(header.Parse(buffer, sizeof(buffer)));
  buffer[0] = 1 << 6;
  EXPECT_FALSE(header.Parse(buffer, sizeof(buffer)));
  buffer[0] = 2 << 6;
  EXPECT_TRUE(header.Parse(buffer, sizeof(buffer)));
  buffer[0] = 3 << 6;
  EXPECT_FALSE(header.Parse(buffer, sizeof(buffer)));
}

TEST(RtcpCommonHeaderTest, PacketSize) {
  uint8_t buffer[] = {0x80, 0x00, 0x00, 0x02, 0x00, 0x00,
                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  CommonHeader header;
  EXPECT_FALSE(header.Parse(buffer, sizeof(buffer) - 1));
  EXPECT_TRUE(header.Parse(buffer, sizeof(buffer)));
  EXPECT_EQ(8u, header.payload_size_bytes());
  EXPECT_EQ(buffer + sizeof(buffer), header.NextPacket());
  EXPECT_EQ(sizeof(buffer), header.packet_size());
}

TEST(RtcpCommonHeaderTest, PaddingAndPayloadSize) {
  // Set v = 2, p = 1, but leave fmt, pt as 0.
  uint8_t buffer[] = {0xa0, 0x00, 0x00, 0x00, 0x00, 0x00,
                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  CommonHeader header;
  // Padding bit set, but no byte for padding (can't specify padding length).
  EXPECT_FALSE(header.Parse(buffer, 4));

  buffer[3] = 2;  //  Set payload size to 2x32bit.
  const size_t kPayloadSizeBytes = buffer[3] * 4;
  const size_t kPaddingAddress =
      CommonHeader::kHeaderSizeBytes + kPayloadSizeBytes - 1;

  // Padding one byte larger than possible.
  buffer[kPaddingAddress] = kPayloadSizeBytes + 1;
  EXPECT_FALSE(header.Parse(buffer, sizeof(buffer)));

  // Invalid zero padding size.
  buffer[kPaddingAddress] = 0;
  EXPECT_FALSE(header.Parse(buffer, sizeof(buffer)));

  // Pure padding packet.
  buffer[kPaddingAddress] = kPayloadSizeBytes;
  EXPECT_TRUE(header.Parse(buffer, sizeof(buffer)));
  EXPECT_EQ(0u, header.payload_size_bytes());
  EXPECT_EQ(buffer + sizeof(buffer), header.NextPacket());
  EXPECT_EQ(header.payload(), buffer + CommonHeader::kHeaderSizeBytes);
  EXPECT_EQ(header.packet_size(), sizeof(buffer));

  // Single byte of actual data.
  buffer[kPaddingAddress] = kPayloadSizeBytes - 1;
  EXPECT_TRUE(header.Parse(buffer, sizeof(buffer)));
  EXPECT_EQ(1u, header.payload_size_bytes());
  EXPECT_EQ(buffer + sizeof(buffer), header.NextPacket());
  EXPECT_EQ(header.packet_size(), sizeof(buffer));
}

TEST(RtcpCommonHeaderTest, FormatAndPayloadType) {
  uint8_t buffer[] = {0x9e, 0xab, 0x00, 0x00};
  CommonHeader header;
  EXPECT_TRUE(header.Parse(buffer, sizeof(buffer)));

  EXPECT_EQ(header.count(), 0x1e);
  EXPECT_EQ(header.fmt(), 0x1e);
  EXPECT_EQ(header.type(), 0xab);
  EXPECT_EQ(header.payload_size_bytes(), 0u);
  EXPECT_EQ(header.payload(), buffer + CommonHeader::kHeaderSizeBytes);
}
}  // namespace webrtc
