/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
#include "rtc_tools/video_file_writer.h"

#include <stdint.h>

#include <cstdio>
#include <string>

#include "api/video/video_frame_buffer.h"
#include "rtc_tools/video_file_reader.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace test {

class VideoFileWriterTest : public ::testing::Test {
 public:
  void SetUp() override {
    video_filename_ =
        TempFilename(webrtc::test::OutputPath(), "test_video_file.y4m");

    // Create simple test video of size 6x4.
    FILE* file = fopen(video_filename_.c_str(), "wb");
    ASSERT_TRUE(file != nullptr);
    fprintf(file, "YUV4MPEG2 W6 H4 F60:1 C420 dummyParam\n");
    fprintf(file, "FRAME\n");

    const int i420_size = width * height * 3 / 2;
    // First frame.
    for (int i = 0; i < i420_size; ++i)
      fputc(static_cast<char>(i), file);
    fprintf(file, "FRAME\n");
    // Second frame.
    for (int i = 0; i < i420_size; ++i)
      fputc(static_cast<char>(i + i420_size), file);
    fclose(file);

    // Open the newly created file.
    video_ = webrtc::test::OpenY4mFile(video_filename_);
    ASSERT_TRUE(video_);
    ASSERT_EQ(video_->number_of_frames(), 2u);
  }

  void TearDown() override {
    if (!video_filename_.empty()) {
      RemoveFile(video_filename_);
    }
    if (!written_video_filename_.empty()) {
      RemoveFile(written_video_filename_);
    }
  }

  // Write and read Y4M file.
  void WriteVideoY4m() {
    // Cleanup existing file if any.
    if (!written_video_filename_.empty()) {
      RemoveFile(written_video_filename_);
    }
    // Create an unique filename, e.g. test_video_file2.y4mZapata.
    written_video_filename_ =
        TempFilename(webrtc::test::OutputPath(), "test_video_file2.y4m");
    webrtc::test::WriteY4mVideoToFile(video_, written_video_filename_, fps);
    written_video_ = webrtc::test::OpenY4mFile(written_video_filename_);
    ASSERT_TRUE(written_video_);
  }

  // Write and read YUV file.
  void WriteVideoYuv() {
    // Cleanup existing file if any.
    if (!written_video_filename_.empty()) {
      RemoveFile(written_video_filename_);
    }
    // Create an unique filename, e.g. test_video_file2.yuvZapata.
    written_video_filename_ =
        TempFilename(webrtc::test::OutputPath(), "test_video_file2.yuv");
    webrtc::test::WriteYuvVideoToFile(video_, written_video_filename_, fps);
    written_video_ =
        webrtc::test::OpenYuvFile(written_video_filename_, width, height);
    ASSERT_TRUE(written_video_);
  }

  const int width = 6;
  const int height = 4;
  const int fps = 60;
  rtc::scoped_refptr<webrtc::test::Video> video_;
  rtc::scoped_refptr<webrtc::test::Video> written_video_;
  // Each video object must be backed by file!
  std::string video_filename_;
  std::string written_video_filename_;
};

TEST_F(VideoFileWriterTest, TestParsingFileHeaderY4m) {
  WriteVideoY4m();
  EXPECT_EQ(video_->width(), written_video_->width());
  EXPECT_EQ(video_->height(), written_video_->height());
}

TEST_F(VideoFileWriterTest, TestParsingFileHeaderYuv) {
  WriteVideoYuv();
  EXPECT_EQ(video_->width(), written_video_->width());
  EXPECT_EQ(video_->height(), written_video_->height());
}

TEST_F(VideoFileWriterTest, TestParsingNumberOfFramesY4m) {
  WriteVideoY4m();
  EXPECT_EQ(video_->number_of_frames(), written_video_->number_of_frames());
}

TEST_F(VideoFileWriterTest, TestParsingNumberOfFramesYuv) {
  WriteVideoYuv();
  EXPECT_EQ(video_->number_of_frames(), written_video_->number_of_frames());
}

TEST_F(VideoFileWriterTest, TestPixelContentY4m) {
  WriteVideoY4m();
  int cnt = 0;
  for (const rtc::scoped_refptr<I420BufferInterface> frame : *written_video_) {
    for (int i = 0; i < width * height; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataY()[i]);
    for (int i = 0; i < width / 2 * height / 2; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataU()[i]);
    for (int i = 0; i < width / 2 * height / 2; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataV()[i]);
  }
}

TEST_F(VideoFileWriterTest, TestPixelContentYuv) {
  WriteVideoYuv();
  int cnt = 0;
  for (const rtc::scoped_refptr<I420BufferInterface> frame : *written_video_) {
    for (int i = 0; i < width * height; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataY()[i]);
    for (int i = 0; i < width / 2 * height / 2; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataU()[i]);
    for (int i = 0; i < width / 2 * height / 2; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataV()[i]);
  }
}

}  // namespace test
}  // namespace webrtc
