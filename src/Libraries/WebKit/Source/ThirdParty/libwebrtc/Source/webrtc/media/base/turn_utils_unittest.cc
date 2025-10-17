/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#include "media/base/turn_utils.h"

#include "test/gtest.h"

namespace cricket {

// Invalid TURN send indication messages. Messages are proper STUN
// messages with incorrect values in attributes.
TEST(TurnUtilsTest, InvalidTurnSendIndicationMessages) {
  size_t content_pos = SIZE_MAX;
  size_t content_size = SIZE_MAX;

  // Stun Indication message with Zero length
  uint8_t kTurnSendIndicationMsgWithNoAttributes[] = {
      0x00, 0x16, 0x00, 0x00,  // Zero length
      0x21, 0x12, 0xA4, 0x42,  // magic cookie
      '0',  '1',  '2',  '3',   // transaction id
      '4',  '5',  '6',  '7',  '8', '9', 'a', 'b',
  };
  EXPECT_FALSE(UnwrapTurnPacket(kTurnSendIndicationMsgWithNoAttributes,
                                sizeof(kTurnSendIndicationMsgWithNoAttributes),
                                &content_pos, &content_size));
  EXPECT_EQ(SIZE_MAX, content_pos);
  EXPECT_EQ(SIZE_MAX, content_size);

  // Stun Send Indication message with invalid length in stun header.
  const uint8_t kTurnSendIndicationMsgWithInvalidLength[] = {
      0x00, 0x16, 0xFF, 0x00,  // length of 0xFF00
      0x21, 0x12, 0xA4, 0x42,  // magic cookie
      '0',  '1',  '2',  '3',   // transaction id
      '4',  '5',  '6',  '7',  '8', '9', 'a', 'b',
  };
  EXPECT_FALSE(UnwrapTurnPacket(kTurnSendIndicationMsgWithInvalidLength,
                                sizeof(kTurnSendIndicationMsgWithInvalidLength),
                                &content_pos, &content_size));
  EXPECT_EQ(SIZE_MAX, content_pos);
  EXPECT_EQ(SIZE_MAX, content_size);

  // Stun Send Indication message with no DATA attribute in message.
  const uint8_t kTurnSendIndicatinMsgWithNoDataAttribute[] = {
      // clang-format off
      // clang formatting doesn't respect inline comments.
      0x00, 0x16, 0x00, 0x08,  // length of
      0x21, 0x12, 0xA4, 0x42,  // magic cookie
      '0',  '1',  '2',  '3',   // transaction id
      '4',  '5',  '6',  '7',  '8',  '9', 'a',  'b',
      0x00, 0x20, 0x00, 0x04,  // Mapped address.
      0x00, 0x00, 0x00, 0x00,
      // clang-format on
  };
  EXPECT_FALSE(
      UnwrapTurnPacket(kTurnSendIndicatinMsgWithNoDataAttribute,
                       sizeof(kTurnSendIndicatinMsgWithNoDataAttribute),
                       &content_pos, &content_size));
  EXPECT_EQ(SIZE_MAX, content_pos);
  EXPECT_EQ(SIZE_MAX, content_size);
}

// Valid TURN Send Indication messages.
TEST(TurnUtilsTest, ValidTurnSendIndicationMessage) {
  size_t content_pos = SIZE_MAX;
  size_t content_size = SIZE_MAX;
  // A valid STUN indication message with a valid RTP header in data attribute
  // payload field and no extension bit set.
  const uint8_t kTurnSendIndicationMsgWithoutRtpExtension[] = {
      // clang-format off
      // clang formatting doesn't respect inline comments.
      0x00, 0x16, 0x00, 0x18,  // length of
      0x21, 0x12, 0xA4, 0x42,  // magic cookie
      '0',  '1',  '2',  '3',   // transaction id
      '4',  '5',  '6',  '7',  '8',  '9',  'a',  'b',
      0x00, 0x20, 0x00, 0x04,  // Mapped address.
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x13, 0x00, 0x0C,  // Data attribute.
      0x80, 0x00, 0x00, 0x00,  // RTP packet.
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      // clang-format on
  };
  EXPECT_TRUE(
      UnwrapTurnPacket(kTurnSendIndicationMsgWithoutRtpExtension,
                       sizeof(kTurnSendIndicationMsgWithoutRtpExtension),
                       &content_pos, &content_size));
  EXPECT_EQ(12U, content_size);
  EXPECT_EQ(32U, content_pos);
}

// Verify that parsing of valid TURN Channel Messages.
TEST(TurnUtilsTest, ValidTurnChannelMessages) {
  const uint8_t kTurnChannelMsgWithRtpPacket[] = {
      // clang-format off
      // clang formatting doesn't respect inline comments.
      0x40, 0x00, 0x00, 0x0C,
      0x80, 0x00, 0x00, 0x00,  // RTP packet.
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      // clang-format on
  };

  size_t content_pos = 0, content_size = 0;
  EXPECT_TRUE(UnwrapTurnPacket(kTurnChannelMsgWithRtpPacket,
                               sizeof(kTurnChannelMsgWithRtpPacket),
                               &content_pos, &content_size));
  EXPECT_EQ(12U, content_size);
  EXPECT_EQ(4U, content_pos);
}

TEST(TurnUtilsTest, ChannelMessageZeroLength) {
  const uint8_t kTurnChannelMsgWithZeroLength[] = {0x40, 0x00, 0x00, 0x00};
  size_t content_pos = SIZE_MAX;
  size_t content_size = SIZE_MAX;
  EXPECT_TRUE(UnwrapTurnPacket(kTurnChannelMsgWithZeroLength,
                               sizeof(kTurnChannelMsgWithZeroLength),
                               &content_pos, &content_size));
  EXPECT_EQ(4u, content_pos);
  EXPECT_EQ(0u, content_size);
}

}  // namespace cricket
