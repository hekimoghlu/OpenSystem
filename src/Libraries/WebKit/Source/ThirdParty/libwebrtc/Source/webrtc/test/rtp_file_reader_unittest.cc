/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
#include "test/rtp_file_reader.h"

#include <map>
#include <memory>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_util.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {

class TestRtpFileReader : public ::testing::Test {
 public:
  void Init(const std::string& filename, bool headers_only_file) {
    std::string filepath =
        test::ResourcePath("video_coding/" + filename, "rtp");
    rtp_packet_source_.reset(
        test::RtpFileReader::Create(test::RtpFileReader::kRtpDump, filepath));
    ASSERT_TRUE(rtp_packet_source_.get() != NULL);
    headers_only_file_ = headers_only_file;
  }

  int CountRtpPackets() {
    test::RtpPacket packet;
    int c = 0;
    while (rtp_packet_source_->NextPacket(&packet)) {
      if (headers_only_file_)
        EXPECT_LT(packet.length, packet.original_length);
      else
        EXPECT_EQ(packet.length, packet.original_length);
      c++;
    }
    return c;
  }

 private:
  std::unique_ptr<test::RtpFileReader> rtp_packet_source_;
  bool headers_only_file_;
};

TEST_F(TestRtpFileReader, Test60Packets) {
  Init("pltype103", false);
  EXPECT_EQ(60, CountRtpPackets());
}

TEST_F(TestRtpFileReader, Test60PacketsHeaderOnly) {
  Init("pltype103_header_only", true);
  EXPECT_EQ(60, CountRtpPackets());
}

typedef std::map<uint32_t, int> PacketsPerSsrc;

class TestPcapFileReader : public ::testing::Test {
 public:
  void Init(const std::string& filename) {
    std::string filepath =
        test::ResourcePath("video_coding/" + filename, "pcap");
    rtp_packet_source_.reset(
        test::RtpFileReader::Create(test::RtpFileReader::kPcap, filepath));
    ASSERT_TRUE(rtp_packet_source_.get() != NULL);
  }

  int CountRtpPackets() {
    int c = 0;
    test::RtpPacket packet;
    while (rtp_packet_source_->NextPacket(&packet)) {
      EXPECT_EQ(packet.length, packet.original_length);
      c++;
    }
    return c;
  }

  PacketsPerSsrc CountRtpPacketsPerSsrc() {
    PacketsPerSsrc pps;
    test::RtpPacket packet;
    while (rtp_packet_source_->NextPacket(&packet)) {
      rtc::ArrayView<const uint8_t> raw(packet.data, packet.length);
      if (IsRtpPacket(raw)) {
        pps[ParseRtpSsrc(raw)]++;
      }
    }
    return pps;
  }

 private:
  std::unique_ptr<test::RtpFileReader> rtp_packet_source_;
};

TEST_F(TestPcapFileReader, TestEthernetIIFrame) {
  Init("frame-ethernet-ii");
  EXPECT_EQ(368, CountRtpPackets());
}

TEST_F(TestPcapFileReader, TestLoopbackFrame) {
  Init("frame-loopback");
  EXPECT_EQ(491, CountRtpPackets());
}

TEST_F(TestPcapFileReader, TestTwoSsrc) {
  Init("ssrcs-2");
  PacketsPerSsrc pps = CountRtpPacketsPerSsrc();
  EXPECT_EQ(2UL, pps.size());
  EXPECT_EQ(370, pps[0x78d48f61]);
  EXPECT_EQ(60, pps[0xae94130b]);
}

TEST_F(TestPcapFileReader, TestThreeSsrc) {
  Init("ssrcs-3");
  PacketsPerSsrc pps = CountRtpPacketsPerSsrc();
  EXPECT_EQ(3UL, pps.size());
  EXPECT_EQ(162, pps[0x938c5eaa]);
  EXPECT_EQ(113, pps[0x59fe6ef0]);
  EXPECT_EQ(61, pps[0xed2bd2ac]);
}
}  // namespace webrtc
