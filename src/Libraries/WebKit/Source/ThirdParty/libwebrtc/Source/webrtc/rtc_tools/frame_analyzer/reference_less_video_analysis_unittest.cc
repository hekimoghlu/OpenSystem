/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#include <vector>

#include "api/scoped_refptr.h"
#include "rtc_tools/frame_analyzer/reference_less_video_analysis_lib.h"
#include "rtc_tools/video_file_reader.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

class ReferenceLessVideoAnalysisTest : public ::testing::Test {
 public:
  void SetUp() override {
    video = webrtc::test::OpenY4mFile(
        webrtc::test::ResourcePath("reference_less_video_test_file", "y4m"));
    ASSERT_TRUE(video);
  }

  rtc::scoped_refptr<webrtc::test::Video> video;
  std::vector<double> psnr_per_frame;
  std::vector<double> ssim_per_frame;
};

TEST_F(ReferenceLessVideoAnalysisTest, MatchComputedMetrics) {
  compute_metrics(video, &psnr_per_frame, &ssim_per_frame);
  EXPECT_EQ(74, (int)psnr_per_frame.size());

  ASSERT_NEAR(27.2f, psnr_per_frame[1], 0.1f);
  ASSERT_NEAR(24.9f, psnr_per_frame[5], 0.1f);

  ASSERT_NEAR(0.9f, ssim_per_frame[1], 0.1f);
  ASSERT_NEAR(0.9f, ssim_per_frame[5], 0.1f);
}

TEST_F(ReferenceLessVideoAnalysisTest, MatchIdenticalFrameClusters) {
  compute_metrics(video, &psnr_per_frame, &ssim_per_frame);
  std::vector<int> identical_frame_clusters =
      find_frame_clusters(psnr_per_frame, ssim_per_frame);
  EXPECT_EQ(5, (int)identical_frame_clusters.size());
  EXPECT_EQ(1, identical_frame_clusters[0]);
  EXPECT_EQ(1, identical_frame_clusters[4]);
}
