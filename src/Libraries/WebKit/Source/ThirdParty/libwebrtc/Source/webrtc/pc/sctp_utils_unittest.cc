/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#include "pc/sctp_utils.h"

#include <stdint.h>

#include <limits>
#include <optional>

#include "api/priority.h"
#include "media/sctp/sctp_transport_internal.h"
#include "rtc_base/byte_buffer.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "test/gtest.h"

using webrtc::StreamId;

class SctpUtilsTest : public ::testing::Test {
 public:
  void VerifyOpenMessageFormat(const rtc::CopyOnWriteBuffer& packet,
                               const std::string& label,
                               const webrtc::DataChannelInit& config) {
    uint8_t message_type;
    uint8_t channel_type;
    uint32_t reliability;
    uint16_t priority;
    uint16_t label_length;
    uint16_t protocol_length;

    rtc::ByteBufferReader buffer(packet);
    ASSERT_TRUE(buffer.ReadUInt8(&message_type));
    EXPECT_EQ(0x03, message_type);

    ASSERT_TRUE(buffer.ReadUInt8(&channel_type));
    if (config.ordered) {
      EXPECT_EQ(
          config.maxRetransmits ? 0x01 : (config.maxRetransmitTime ? 0x02 : 0),
          channel_type);
    } else {
      EXPECT_EQ(config.maxRetransmits
                    ? 0x81
                    : (config.maxRetransmitTime ? 0x82 : 0x80),
                channel_type);
    }

    ASSERT_TRUE(buffer.ReadUInt16(&priority));
    if (config.priority) {
      // Exact values are checked by round-trip conversion, but
      // all values defined are greater than zero.
      EXPECT_EQ(priority, config.priority->value());
    } else {
      EXPECT_EQ(priority,
                webrtc::PriorityValue(webrtc::Priority::kLow).value());
    }

    ASSERT_TRUE(buffer.ReadUInt32(&reliability));
    if (config.maxRetransmits || config.maxRetransmitTime) {
      EXPECT_EQ(config.maxRetransmits ? *config.maxRetransmits
                                      : *config.maxRetransmitTime,
                static_cast<int>(reliability));
    }

    ASSERT_TRUE(buffer.ReadUInt16(&label_length));
    ASSERT_TRUE(buffer.ReadUInt16(&protocol_length));
    EXPECT_EQ(label.size(), label_length);
    EXPECT_EQ(config.protocol.size(), protocol_length);

    absl::string_view label_output;
    ASSERT_TRUE(buffer.ReadStringView(&label_output, label_length));
    EXPECT_EQ(label, label_output);
    absl::string_view protocol_output;
    ASSERT_TRUE(buffer.ReadStringView(&protocol_output, protocol_length));
    EXPECT_EQ(config.protocol, protocol_output);
  }
};

TEST_F(SctpUtilsTest, WriteParseOpenMessageWithOrderedReliable) {
  webrtc::DataChannelInit config;
  std::string label = "abc";
  config.protocol = "y";

  rtc::CopyOnWriteBuffer packet;
  ASSERT_TRUE(webrtc::WriteDataChannelOpenMessage(label, config, &packet));

  VerifyOpenMessageFormat(packet, label, config);

  std::string output_label;
  webrtc::DataChannelInit output_config;
  ASSERT_TRUE(webrtc::ParseDataChannelOpenMessage(packet, &output_label,
                                                  &output_config));

  EXPECT_EQ(label, output_label);
  EXPECT_EQ(config.protocol, output_config.protocol);
  EXPECT_EQ(config.ordered, output_config.ordered);
  EXPECT_EQ(config.maxRetransmitTime, output_config.maxRetransmitTime);
  EXPECT_EQ(config.maxRetransmits, output_config.maxRetransmits);
}

TEST_F(SctpUtilsTest, WriteParseOpenMessageWithMaxRetransmitTime) {
  webrtc::DataChannelInit config;
  std::string label = "abc";
  config.ordered = false;
  config.maxRetransmitTime = 10;
  config.protocol = "y";

  rtc::CopyOnWriteBuffer packet;
  ASSERT_TRUE(webrtc::WriteDataChannelOpenMessage(label, config, &packet));

  VerifyOpenMessageFormat(packet, label, config);

  std::string output_label;
  webrtc::DataChannelInit output_config;
  ASSERT_TRUE(webrtc::ParseDataChannelOpenMessage(packet, &output_label,
                                                  &output_config));

  EXPECT_EQ(label, output_label);
  EXPECT_EQ(config.protocol, output_config.protocol);
  EXPECT_EQ(config.ordered, output_config.ordered);
  EXPECT_EQ(*config.maxRetransmitTime, *output_config.maxRetransmitTime);
  EXPECT_FALSE(output_config.maxRetransmits);
}

TEST_F(SctpUtilsTest, WriteParseOpenMessageWithMaxRetransmits) {
  webrtc::DataChannelInit config;
  std::string label = "abc";
  config.maxRetransmits = 10;
  config.protocol = "y";

  rtc::CopyOnWriteBuffer packet;
  ASSERT_TRUE(webrtc::WriteDataChannelOpenMessage(label, config, &packet));

  VerifyOpenMessageFormat(packet, label, config);

  std::string output_label;
  webrtc::DataChannelInit output_config;
  ASSERT_TRUE(webrtc::ParseDataChannelOpenMessage(packet, &output_label,
                                                  &output_config));

  EXPECT_EQ(label, output_label);
  EXPECT_EQ(config.protocol, output_config.protocol);
  EXPECT_EQ(config.ordered, output_config.ordered);
  EXPECT_EQ(config.maxRetransmits, output_config.maxRetransmits);
  EXPECT_FALSE(output_config.maxRetransmitTime);
}

TEST_F(SctpUtilsTest, WriteParseOpenMessageWithPriority) {
  webrtc::DataChannelInit config;
  std::string label = "abc";
  config.protocol = "y";
  config.priority = webrtc::PriorityValue(webrtc::Priority::kVeryLow);

  rtc::CopyOnWriteBuffer packet;
  ASSERT_TRUE(webrtc::WriteDataChannelOpenMessage(label, config, &packet));

  VerifyOpenMessageFormat(packet, label, config);

  std::string output_label;
  webrtc::DataChannelInit output_config;
  ASSERT_TRUE(webrtc::ParseDataChannelOpenMessage(packet, &output_label,
                                                  &output_config));

  EXPECT_EQ(label, output_label);
  ASSERT_TRUE(output_config.priority);
  EXPECT_EQ(*config.priority, *output_config.priority);
}

TEST_F(SctpUtilsTest, WriteParseAckMessage) {
  rtc::CopyOnWriteBuffer packet;
  webrtc::WriteDataChannelOpenAckMessage(&packet);

  uint8_t message_type;
  rtc::ByteBufferReader buffer(packet);
  ASSERT_TRUE(buffer.ReadUInt8(&message_type));
  EXPECT_EQ(0x02, message_type);

  EXPECT_TRUE(webrtc::ParseDataChannelOpenAckMessage(packet));
}

TEST_F(SctpUtilsTest, TestIsOpenMessage) {
  rtc::CopyOnWriteBuffer open(1);
  open.MutableData()[0] = 0x03;
  EXPECT_TRUE(webrtc::IsOpenMessage(open));

  rtc::CopyOnWriteBuffer openAck(1);
  openAck.MutableData()[0] = 0x02;
  EXPECT_FALSE(webrtc::IsOpenMessage(openAck));

  rtc::CopyOnWriteBuffer invalid(1);
  invalid.MutableData()[0] = 0x01;
  EXPECT_FALSE(webrtc::IsOpenMessage(invalid));

  rtc::CopyOnWriteBuffer empty;
  EXPECT_FALSE(webrtc::IsOpenMessage(empty));
}

TEST(SctpSidTest, Basics) {
  // These static asserts are mostly here to aid with readability (i.e. knowing
  // what these constants represent).
  static_assert(cricket::kMinSctpSid == 0, "Min stream id should be 0");
  static_assert(cricket::kMaxSctpSid <= cricket::kSpecMaxSctpSid, "");
  static_assert(
      cricket::kSpecMaxSctpSid == std::numeric_limits<uint16_t>::max(),
      "Max legal sctp stream value should be 0xffff");
}
