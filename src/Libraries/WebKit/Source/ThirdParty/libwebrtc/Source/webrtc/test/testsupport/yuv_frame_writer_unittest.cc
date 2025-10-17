/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <string>

#include "test/gtest.h"
#include "test/testsupport/file_utils.h"
#include "test/testsupport/frame_writer.h"

namespace webrtc {
namespace test {

namespace {
const size_t kFrameWidth = 50;
const size_t kFrameHeight = 20;
const size_t kFrameLength = 3 * kFrameWidth * kFrameHeight / 2;  // I420.
}  // namespace

class YuvFrameWriterTest : public ::testing::Test {
 protected:
  YuvFrameWriterTest() = default;
  ~YuvFrameWriterTest() override = default;

  void SetUp() override {
    temp_filename_ = webrtc::test::TempFilename(webrtc::test::OutputPath(),
                                                "yuv_frame_writer_unittest");
    frame_writer_.reset(
        new YuvFrameWriterImpl(temp_filename_, kFrameWidth, kFrameHeight));
    ASSERT_TRUE(frame_writer_->Init());
  }

  void TearDown() override { remove(temp_filename_.c_str()); }

  std::unique_ptr<FrameWriter> frame_writer_;
  std::string temp_filename_;
};

TEST_F(YuvFrameWriterTest, InitSuccess) {}

TEST_F(YuvFrameWriterTest, FrameLength) {
  EXPECT_EQ(kFrameLength, frame_writer_->FrameLength());
}

TEST_F(YuvFrameWriterTest, WriteFrame) {
  uint8_t buffer[kFrameLength];
  memset(buffer, 9, kFrameLength);  // Write lots of 9s to the buffer.
  bool result = frame_writer_->WriteFrame(buffer);
  ASSERT_TRUE(result);

  frame_writer_->Close();
  EXPECT_EQ(kFrameLength, GetFileSize(temp_filename_));
}

TEST_F(YuvFrameWriterTest, WriteFrameUninitialized) {
  uint8_t buffer[kFrameLength];
  YuvFrameWriterImpl frame_writer(temp_filename_, kFrameWidth, kFrameHeight);
  EXPECT_FALSE(frame_writer.WriteFrame(buffer));
}

}  // namespace test
}  // namespace webrtc
