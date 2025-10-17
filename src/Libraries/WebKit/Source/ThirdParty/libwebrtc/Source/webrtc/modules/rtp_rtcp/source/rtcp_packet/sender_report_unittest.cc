/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#include "modules/rtp_rtcp/source/rtcp_packet/sender_report.h"

#include <utility>

#include "test/gmock.h"
#include "test/gtest.h"
#include "test/rtcp_packet_parser.h"

using ::testing::ElementsAreArray;
using ::testing::make_tuple;
using webrtc::rtcp::ReportBlock;
using webrtc::rtcp::SenderReport;

namespace webrtc {
namespace {
const uint32_t kSenderSsrc = 0x12345678;
const uint32_t kRemoteSsrc = 0x23456789;
const NtpTime kNtp(0x11121418, 0x22242628);
const uint32_t kRtpTimestamp = 0x33343536;
const uint32_t kPacketCount = 0x44454647;
const uint32_t kOctetCount = 0x55565758;
const uint8_t kPacket[] = {0x80, 200,  0x00, 0x06, 0x12, 0x34, 0x56,
                           0x78, 0x11, 0x12, 0x14, 0x18, 0x22, 0x24,
                           0x26, 0x28, 0x33, 0x34, 0x35, 0x36, 0x44,
                           0x45, 0x46, 0x47, 0x55, 0x56, 0x57, 0x58};
}  // namespace

TEST(RtcpPacketSenderReportTest, CreateWithoutReportBlocks) {
  SenderReport sr;
  sr.SetSenderSsrc(kSenderSsrc);
  sr.SetNtp(kNtp);
  sr.SetRtpTimestamp(kRtpTimestamp);
  sr.SetPacketCount(kPacketCount);
  sr.SetOctetCount(kOctetCount);

  rtc::Buffer raw = sr.Build();
  EXPECT_THAT(make_tuple(raw.data(), raw.size()), ElementsAreArray(kPacket));
}

TEST(RtcpPacketSenderReportTest, ParseWithoutReportBlocks) {
  SenderReport parsed;
  EXPECT_TRUE(test::ParseSinglePacket(kPacket, &parsed));

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  EXPECT_EQ(kNtp, parsed.ntp());
  EXPECT_EQ(kRtpTimestamp, parsed.rtp_timestamp());
  EXPECT_EQ(kPacketCount, parsed.sender_packet_count());
  EXPECT_EQ(kOctetCount, parsed.sender_octet_count());
  EXPECT_TRUE(parsed.report_blocks().empty());
}

TEST(RtcpPacketSenderReportTest, CreateAndParseWithOneReportBlock) {
  ReportBlock rb;
  rb.SetMediaSsrc(kRemoteSsrc);

  SenderReport sr;
  sr.SetSenderSsrc(kSenderSsrc);
  EXPECT_TRUE(sr.AddReportBlock(rb));

  rtc::Buffer raw = sr.Build();
  SenderReport parsed;
  EXPECT_TRUE(test::ParseSinglePacket(raw, &parsed));

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  EXPECT_EQ(1u, parsed.report_blocks().size());
  EXPECT_EQ(kRemoteSsrc, parsed.report_blocks()[0].source_ssrc());
}

TEST(RtcpPacketSenderReportTest, CreateAndParseWithTwoReportBlocks) {
  ReportBlock rb1;
  rb1.SetMediaSsrc(kRemoteSsrc);
  ReportBlock rb2;
  rb2.SetMediaSsrc(kRemoteSsrc + 1);

  SenderReport sr;
  sr.SetSenderSsrc(kSenderSsrc);
  EXPECT_TRUE(sr.AddReportBlock(rb1));
  EXPECT_TRUE(sr.AddReportBlock(rb2));

  rtc::Buffer raw = sr.Build();
  SenderReport parsed;
  EXPECT_TRUE(test::ParseSinglePacket(raw, &parsed));

  EXPECT_EQ(kSenderSsrc, parsed.sender_ssrc());
  EXPECT_EQ(2u, parsed.report_blocks().size());
  EXPECT_EQ(kRemoteSsrc, parsed.report_blocks()[0].source_ssrc());
  EXPECT_EQ(kRemoteSsrc + 1, parsed.report_blocks()[1].source_ssrc());
}

TEST(RtcpPacketSenderReportTest, CreateWithTooManyReportBlocks) {
  SenderReport sr;
  sr.SetSenderSsrc(kSenderSsrc);
  ReportBlock rb;
  for (size_t i = 0; i < SenderReport::kMaxNumberOfReportBlocks; ++i) {
    rb.SetMediaSsrc(kRemoteSsrc + i);
    EXPECT_TRUE(sr.AddReportBlock(rb));
  }
  rb.SetMediaSsrc(kRemoteSsrc + SenderReport::kMaxNumberOfReportBlocks);
  EXPECT_FALSE(sr.AddReportBlock(rb));
}

TEST(RtcpPacketSenderReportTest, SetReportBlocksOverwritesOldBlocks) {
  SenderReport sr;
  ReportBlock report_block;
  // Use jitter field of the report blocks to distinguish them.
  report_block.SetJitter(1001u);
  sr.AddReportBlock(report_block);
  ASSERT_EQ(sr.report_blocks().size(), 1u);
  ASSERT_EQ(sr.report_blocks()[0].jitter(), 1001u);

  std::vector<ReportBlock> blocks(3u);
  blocks[0].SetJitter(2001u);
  blocks[1].SetJitter(3001u);
  blocks[2].SetJitter(4001u);
  EXPECT_TRUE(sr.SetReportBlocks(blocks));
  ASSERT_EQ(sr.report_blocks().size(), 3u);
  EXPECT_EQ(sr.report_blocks()[0].jitter(), 2001u);
  EXPECT_EQ(sr.report_blocks()[1].jitter(), 3001u);
  EXPECT_EQ(sr.report_blocks()[2].jitter(), 4001u);
}

TEST(RtcpPacketSenderReportTest, SetReportBlocksMaxLimit) {
  SenderReport sr;
  std::vector<ReportBlock> max_blocks(SenderReport::kMaxNumberOfReportBlocks);
  EXPECT_TRUE(sr.SetReportBlocks(std::move(max_blocks)));

  std::vector<ReportBlock> one_too_many_blocks(
      SenderReport::kMaxNumberOfReportBlocks + 1);
  EXPECT_FALSE(sr.SetReportBlocks(std::move(one_too_many_blocks)));
}

}  // namespace webrtc
