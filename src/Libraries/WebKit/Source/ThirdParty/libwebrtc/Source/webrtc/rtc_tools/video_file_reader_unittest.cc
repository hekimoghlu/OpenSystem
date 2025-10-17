/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#include "rtc_tools/video_file_reader.h"

#include <stdint.h>

#include <string>

#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace test {

class Y4mFileReaderTest : public ::testing::Test {
 public:
  void SetUp() override {
    const std::string filename =
        TempFilename(webrtc::test::OutputPath(), "test_video_file.y4m");

    // Create simple test video of size 6x4.
    FILE* file = fopen(filename.c_str(), "wb");
    ASSERT_TRUE(file != nullptr);
    fprintf(file, "YUV4MPEG2 W6 H4 F57:10 C420 dummyParam\n");
    fprintf(file, "FRAME\n");

    const int width = 6;
    const int height = 4;
    const int i40_size = width * height * 3 / 2;
    // First frame.
    for (int i = 0; i < i40_size; ++i)
      fputc(static_cast<char>(i), file);
    fprintf(file, "FRAME\n");
    // Second frame.
    for (int i = 0; i < i40_size; ++i)
      fputc(static_cast<char>(i + i40_size), file);
    fclose(file);

    // Open the newly created file.
    video = webrtc::test::OpenY4mFile(filename);
    ASSERT_TRUE(video);
  }

  rtc::scoped_refptr<webrtc::test::Video> video;
};

TEST_F(Y4mFileReaderTest, TestParsingFileHeader) {
  EXPECT_EQ(6, video->width());
  EXPECT_EQ(4, video->height());
}

TEST_F(Y4mFileReaderTest, TestParsingNumberOfFrames) {
  EXPECT_EQ(2u, video->number_of_frames());
}

TEST_F(Y4mFileReaderTest, TestPixelContent) {
  int cnt = 0;
  for (const rtc::scoped_refptr<I420BufferInterface> frame : *video) {
    for (int i = 0; i < 6 * 4; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataY()[i]);
    for (int i = 0; i < 3 * 2; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataU()[i]);
    for (int i = 0; i < 3 * 2; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataV()[i]);
  }
}

class YuvFileReaderTest : public ::testing::Test {
 public:
  void SetUp() override {
    const std::string filename =
        TempFilename(webrtc::test::OutputPath(), "test_video_file.yuv");

    // Create simple test video of size 6x4.
    FILE* file = fopen(filename.c_str(), "wb");
    ASSERT_TRUE(file != nullptr);

    const int width = 6;
    const int height = 4;
    const int i40_size = width * height * 3 / 2;
    // First frame.
    for (int i = 0; i < i40_size; ++i)
      fputc(static_cast<char>(i), file);
    // Second frame.
    for (int i = 0; i < i40_size; ++i)
      fputc(static_cast<char>(i + i40_size), file);
    fclose(file);

    // Open the newly created file.
    video = webrtc::test::OpenYuvFile(filename, 6, 4);
    ASSERT_TRUE(video);
  }

  rtc::scoped_refptr<webrtc::test::Video> video;
};

TEST_F(YuvFileReaderTest, TestParsingFileHeader) {
  EXPECT_EQ(6, video->width());
  EXPECT_EQ(4, video->height());
}

TEST_F(YuvFileReaderTest, TestParsingNumberOfFrames) {
  EXPECT_EQ(2u, video->number_of_frames());
}

TEST_F(YuvFileReaderTest, TestPixelContent) {
  int cnt = 0;
  for (const rtc::scoped_refptr<I420BufferInterface> frame : *video) {
    for (int i = 0; i < 6 * 4; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataY()[i]);
    for (int i = 0; i < 3 * 2; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataU()[i]);
    for (int i = 0; i < 3 * 2; ++i, ++cnt)
      EXPECT_EQ(cnt, frame->DataV()[i]);
  }
}

}  // namespace test
}  // namespace webrtc
