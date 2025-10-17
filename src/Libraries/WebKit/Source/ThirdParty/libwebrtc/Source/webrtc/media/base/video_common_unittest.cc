/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#include "media/base/video_common.h"

#include "test/gtest.h"

namespace cricket {

TEST(VideoCommonTest, TestCanonicalFourCC) {
  // Canonical fourccs are not changed.
  EXPECT_EQ(FOURCC_I420, CanonicalFourCC(FOURCC_I420));
  // The special FOURCC_ANY value is not changed.
  EXPECT_EQ(FOURCC_ANY, CanonicalFourCC(FOURCC_ANY));
  // Aliases are translated to the canonical equivalent.
  EXPECT_EQ(FOURCC_I420, CanonicalFourCC(FOURCC_IYUV));
  EXPECT_EQ(FOURCC_I422, CanonicalFourCC(FOURCC_YU16));
  EXPECT_EQ(FOURCC_I444, CanonicalFourCC(FOURCC_YU24));
  EXPECT_EQ(FOURCC_YUY2, CanonicalFourCC(FOURCC_YUYV));
  EXPECT_EQ(FOURCC_YUY2, CanonicalFourCC(FOURCC_YUVS));
  EXPECT_EQ(FOURCC_UYVY, CanonicalFourCC(FOURCC_HDYC));
  EXPECT_EQ(FOURCC_UYVY, CanonicalFourCC(FOURCC_2VUY));
  EXPECT_EQ(FOURCC_MJPG, CanonicalFourCC(FOURCC_JPEG));
  EXPECT_EQ(FOURCC_MJPG, CanonicalFourCC(FOURCC_DMB1));
  EXPECT_EQ(FOURCC_BGGR, CanonicalFourCC(FOURCC_BA81));
  EXPECT_EQ(FOURCC_RAW, CanonicalFourCC(FOURCC_RGB3));
  EXPECT_EQ(FOURCC_24BG, CanonicalFourCC(FOURCC_BGR3));
  EXPECT_EQ(FOURCC_BGRA, CanonicalFourCC(FOURCC_CM32));
  EXPECT_EQ(FOURCC_RAW, CanonicalFourCC(FOURCC_CM24));
}

// Test conversion between interval and fps
TEST(VideoCommonTest, TestVideoFormatFps) {
  EXPECT_EQ(VideoFormat::kMinimumInterval, VideoFormat::FpsToInterval(0));
  EXPECT_EQ(rtc::kNumNanosecsPerSec / 20, VideoFormat::FpsToInterval(20));
  EXPECT_EQ(20, VideoFormat::IntervalToFps(rtc::kNumNanosecsPerSec / 20));
  EXPECT_EQ(0, VideoFormat::IntervalToFps(0));
}

// Test IsSize0x0
TEST(VideoCommonTest, TestVideoFormatIsSize0x0) {
  VideoFormat format;
  EXPECT_TRUE(format.IsSize0x0());
  format.width = 320;
  EXPECT_FALSE(format.IsSize0x0());
}

// Test ToString: print fourcc when it is printable.
TEST(VideoCommonTest, TestVideoFormatToString) {
  VideoFormat format;
  EXPECT_EQ("0x0x0", format.ToString());

  format.fourcc = FOURCC_I420;
  format.width = 640;
  format.height = 480;
  format.interval = VideoFormat::FpsToInterval(20);
  EXPECT_EQ("I420 640x480x20", format.ToString());

  format.fourcc = FOURCC_ANY;
  format.width = 640;
  format.height = 480;
  format.interval = VideoFormat::FpsToInterval(20);
  EXPECT_EQ("640x480x20", format.ToString());
}

// Test comparison.
TEST(VideoCommonTest, TestVideoFormatCompare) {
  VideoFormat format(640, 480, VideoFormat::FpsToInterval(20), FOURCC_I420);
  VideoFormat format2;
  EXPECT_NE(format, format2);

  // Same pixelrate, different fourcc.
  format2 = format;
  format2.fourcc = FOURCC_YUY2;
  EXPECT_NE(format, format2);
  EXPECT_FALSE(format.IsPixelRateLess(format2) ||
               format2.IsPixelRateLess(format2));

  format2 = format;
  format2.interval /= 2;
  EXPECT_TRUE(format.IsPixelRateLess(format2));

  format2 = format;
  format2.width *= 2;
  EXPECT_TRUE(format.IsPixelRateLess(format2));
}

}  // namespace cricket
