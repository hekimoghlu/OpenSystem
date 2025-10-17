/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#include "test/testsupport/video_frame_writer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/test/video/video_frame_writer.h"
#include "api/video/i420_buffer.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"
#include "test/testsupport/frame_reader.h"

namespace webrtc {
namespace test {
namespace {

const size_t kFrameWidth = 50;
const size_t kFrameHeight = 20;
const size_t kFrameLength = 3 * kFrameWidth * kFrameHeight / 2;  // I420.
const size_t kFrameRate = 30;

// Size of header: "YUV4MPEG2 W50 H20 F30:1 C420\n"
const size_t kFileHeaderSize = 29;
// Size of header: "FRAME\n"
const size_t kFrameHeaderSize = 6;

rtc::scoped_refptr<I420Buffer> CreateI420Buffer(int width, int height) {
  rtc::scoped_refptr<I420Buffer> buffer(I420Buffer::Create(width, height));
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      buffer->MutableDataY()[x + y * width] = 128;
    }
  }
  int chroma_width = buffer->ChromaWidth();
  int chroma_height = buffer->ChromaHeight();
  for (int x = 0; x < chroma_width; x++) {
    for (int y = 0; y < chroma_height; y++) {
      buffer->MutableDataU()[x + y * chroma_width] = 1;
      buffer->MutableDataV()[x + y * chroma_width] = 255;
    }
  }
  return buffer;
}

void AssertI420BuffersEq(
    rtc::scoped_refptr<webrtc::I420BufferInterface> actual,
    rtc::scoped_refptr<webrtc::I420BufferInterface> expected) {
  ASSERT_TRUE(actual);

  ASSERT_EQ(actual->width(), expected->width());
  ASSERT_EQ(actual->height(), expected->height());
  const int width = expected->width();
  const int height = expected->height();
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      ASSERT_EQ(actual->DataY()[x + y * width],
                expected->DataY()[x + y * width]);
    }
  }

  ASSERT_EQ(actual->ChromaWidth(), expected->ChromaWidth());
  ASSERT_EQ(actual->ChromaHeight(), expected->ChromaHeight());
  int chroma_width = expected->ChromaWidth();
  int chroma_height = expected->ChromaHeight();
  for (int x = 0; x < chroma_width; x++) {
    for (int y = 0; y < chroma_height; y++) {
      ASSERT_EQ(actual->DataU()[x + y * chroma_width],
                expected->DataU()[x + y * chroma_width]);
      ASSERT_EQ(actual->DataV()[x + y * chroma_width],
                expected->DataV()[x + y * chroma_width]);
    }
  }
}

}  // namespace

class VideoFrameWriterTest : public ::testing::Test {
 protected:
  VideoFrameWriterTest() = default;
  ~VideoFrameWriterTest() override = default;

  void SetUp() override {
    temp_filename_ = webrtc::test::TempFilename(webrtc::test::OutputPath(),
                                                "video_frame_writer_unittest");
    frame_writer_ = CreateFrameWriter();
  }

  virtual std::unique_ptr<VideoFrameWriter> CreateFrameWriter() = 0;

  void TearDown() override { remove(temp_filename_.c_str()); }

  std::unique_ptr<VideoFrameWriter> frame_writer_;
  std::string temp_filename_;
};

class Y4mVideoFrameWriterTest : public VideoFrameWriterTest {
 protected:
  std::unique_ptr<VideoFrameWriter> CreateFrameWriter() override {
    return std::make_unique<Y4mVideoFrameWriterImpl>(
        temp_filename_, kFrameWidth, kFrameHeight, kFrameRate);
  }
};

class YuvVideoFrameWriterTest : public VideoFrameWriterTest {
 protected:
  std::unique_ptr<VideoFrameWriter> CreateFrameWriter() override {
    return std::make_unique<YuvVideoFrameWriterImpl>(temp_filename_,
                                                     kFrameWidth, kFrameHeight);
  }
};

TEST_F(Y4mVideoFrameWriterTest, InitSuccess) {}

TEST_F(Y4mVideoFrameWriterTest, WriteFrame) {
  rtc::scoped_refptr<I420Buffer> expected_buffer =
      CreateI420Buffer(kFrameWidth, kFrameHeight);

  VideoFrame frame =
      VideoFrame::Builder().set_video_frame_buffer(expected_buffer).build();

  ASSERT_TRUE(frame_writer_->WriteFrame(frame));
  ASSERT_TRUE(frame_writer_->WriteFrame(frame));

  frame_writer_->Close();
  EXPECT_EQ(kFileHeaderSize + 2 * kFrameHeaderSize + 2 * kFrameLength,
            GetFileSize(temp_filename_));

  std::unique_ptr<FrameReader> frame_reader =
      CreateY4mFrameReader(temp_filename_);
  AssertI420BuffersEq(frame_reader->PullFrame(), expected_buffer);
  AssertI420BuffersEq(frame_reader->PullFrame(), expected_buffer);
  EXPECT_FALSE(frame_reader->PullFrame());  // End of file.
}

TEST_F(YuvVideoFrameWriterTest, InitSuccess) {}

TEST_F(YuvVideoFrameWriterTest, WriteFrame) {
  rtc::scoped_refptr<I420Buffer> expected_buffer =
      CreateI420Buffer(kFrameWidth, kFrameHeight);

  VideoFrame frame =
      VideoFrame::Builder().set_video_frame_buffer(expected_buffer).build();

  ASSERT_TRUE(frame_writer_->WriteFrame(frame));
  ASSERT_TRUE(frame_writer_->WriteFrame(frame));

  frame_writer_->Close();
  EXPECT_EQ(2 * kFrameLength, GetFileSize(temp_filename_));

  std::unique_ptr<FrameReader> frame_reader = CreateYuvFrameReader(
      temp_filename_,
      Resolution({.width = kFrameWidth, .height = kFrameHeight}));
  AssertI420BuffersEq(frame_reader->PullFrame(), expected_buffer);
  AssertI420BuffersEq(frame_reader->PullFrame(), expected_buffer);
  EXPECT_FALSE(frame_reader->PullFrame());  // End of file.
}

}  // namespace test
}  // namespace webrtc
