/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include "test/rtp_file_writer.h"

#include <stdint.h>
#include <string.h>

#include <memory>

#include "test/gtest.h"
#include "test/rtp_file_reader.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {

class RtpFileWriterTest : public ::testing::Test {
 public:
  void Init(const std::string& filename) {
    filename_ = test::OutputPath() + filename;
    rtp_writer_.reset(
        test::RtpFileWriter::Create(test::RtpFileWriter::kRtpDump, filename_));
  }

  void WriteRtpPackets(int num_packets, int time_ms_offset = 0) {
    ASSERT_TRUE(rtp_writer_.get() != NULL);
    test::RtpPacket packet;
    for (int i = 1; i <= num_packets; ++i) {
      packet.length = i;
      packet.original_length = i;
      packet.time_ms = i + time_ms_offset;
      memset(packet.data, i, packet.length);
      EXPECT_TRUE(rtp_writer_->WritePacket(&packet));
    }
  }

  void CloseOutputFile() { rtp_writer_.reset(); }

  void VerifyFileContents(int expected_packets) {
    ASSERT_TRUE(rtp_writer_.get() == NULL)
        << "Must call CloseOutputFile before VerifyFileContents";
    std::unique_ptr<test::RtpFileReader> rtp_reader(
        test::RtpFileReader::Create(test::RtpFileReader::kRtpDump, filename_));
    ASSERT_TRUE(rtp_reader.get() != NULL);
    test::RtpPacket packet;
    int i = 0;
    while (rtp_reader->NextPacket(&packet)) {
      ++i;
      EXPECT_EQ(static_cast<size_t>(i), packet.length);
      EXPECT_EQ(static_cast<size_t>(i), packet.original_length);
      EXPECT_EQ(static_cast<uint32_t>(i - 1), packet.time_ms);
      for (int j = 0; j < i; ++j) {
        EXPECT_EQ(i, packet.data[j]);
      }
    }
    EXPECT_EQ(expected_packets, i);
  }

 private:
  std::unique_ptr<test::RtpFileWriter> rtp_writer_;
  std::string filename_;
};

TEST_F(RtpFileWriterTest, WriteToRtpDump) {
  Init("test_rtp_file_writer.rtp");
  WriteRtpPackets(10);
  CloseOutputFile();
  VerifyFileContents(10);
}

TEST_F(RtpFileWriterTest, WriteToRtpDumpWithOffset) {
  Init("test_rtp_file_writer.rtp");
  WriteRtpPackets(10, 100);
  CloseOutputFile();
  VerifyFileContents(10);
}

}  // namespace webrtc
