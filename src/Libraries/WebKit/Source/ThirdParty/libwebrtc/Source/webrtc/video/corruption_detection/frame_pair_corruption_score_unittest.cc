/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include "video/corruption_detection/frame_pair_corruption_score.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "api/scoped_refptr.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame_buffer.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"
#include "test/testsupport/frame_reader.h"

namespace webrtc {
namespace {

using test::FrameReader;

// Input video.
constexpr absl::string_view kFilename = "ConferenceMotion_1280_720_50";
constexpr int kWidth = 1280;
constexpr int kHeight = 720;

constexpr absl::string_view kCodecName = "VP8";

// Scale function parameters.
constexpr float kScaleFactor = 14;

// Logistic function parameters.
constexpr float kGrowthRate = 0.5;
constexpr float kMidpoint = 3;

std::unique_ptr<FrameReader> GetFrameGenerator() {
  std::string clip_path = test::ResourcePath(kFilename, "yuv");
  EXPECT_TRUE(test::FileExists(clip_path));
  return CreateYuvFrameReader(clip_path, {.width = kWidth, .height = kHeight},
                              test::YuvFrameReaderImpl::RepeatMode::kPingPong);
}

scoped_refptr<I420BufferInterface> GetDowscaledFrame(
    scoped_refptr<I420BufferInterface> frame,
    float downscale_factor) {
  scoped_refptr<I420Buffer> downscaled_frame =
      I420Buffer::Create(kWidth * downscale_factor, kHeight * downscale_factor);
  downscaled_frame->ScaleFrom(*frame);
  return downscaled_frame;
}

TEST(FramePairCorruptionScorerTest, SameFrameReturnsNoCorruptionScaleFunction) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kScaleFactor, std::nullopt);
  EXPECT_LT(
      frame_pair_corruption_score.CalculateScore(/*qp=*/1, *frame, *frame),
      0.5);
}

TEST(FramePairCorruptionScorerTest,
     SameFrameReturnsNoCorruptionLogisticFunction) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kGrowthRate, kMidpoint, std::nullopt);
  EXPECT_LT(
      frame_pair_corruption_score.CalculateScore(/*qp=*/1, *frame, *frame),
      0.5);
}

TEST(FramePairCorruptionScorerTest,
     HalfScaledFrameReturnsNoCorruptionScaleFunction) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kScaleFactor, std::nullopt);
  EXPECT_LT(frame_pair_corruption_score.CalculateScore(
                /*qp=*/1, *frame,
                *GetDowscaledFrame(frame, /*downscale_factor=*/0.5)),
            0.5);
}

TEST(FramePairCorruptionScorerTest,
     HalfScaledFrameReturnsNoCorruptionLogisticFunction) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kGrowthRate, kMidpoint, std::nullopt);
  EXPECT_LT(frame_pair_corruption_score.CalculateScore(
                /*qp=*/1, *frame,
                *GetDowscaledFrame(frame, /*downscale_factor=*/0.5)),
            0.5);
}

TEST(FramePairCorruptionScorerTest, QuarterScaledFrameReturnsNoCorruption) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kScaleFactor, std::nullopt);
  EXPECT_LT(frame_pair_corruption_score.CalculateScore(
                /*qp=*/1, *frame,
                *GetDowscaledFrame(frame, /*downscale_factor=*/0.25)),
            0.5);
}

TEST(FramePairCorruptionScorerTest,
     DifferentFrameResultsInCorruptionScaleFunction) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  // Get frame number 5, which should be different from the first frame, and
  // hence, indicate a corruption.
  scoped_refptr<I420Buffer> different_frame =
      frame_reader->ReadFrame(/*frame_num=*/5);

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kScaleFactor, std::nullopt);
  EXPECT_GT(frame_pair_corruption_score.CalculateScore(/*qp=*/1, *frame,
                                                       *different_frame),
            0.5);
}

TEST(FramePairCorruptionScorerTest,
     DifferentFrameResultsInCorruptionLogisticFunction) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  // Get frame number 5, which should be different from the first frame, and
  // hence, indicate a corruption.
  scoped_refptr<I420Buffer> different_frame =
      frame_reader->ReadFrame(/*frame_num=*/5);

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kGrowthRate, kMidpoint, std::nullopt);
  EXPECT_GT(frame_pair_corruption_score.CalculateScore(/*qp=*/1, *frame,
                                                       *different_frame),
            0.5);
}

TEST(FramePairCorruptionScorerTest,
     HalfScaledDifferentFrameResultsInCorruptionScaleFunction) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  // Get frame number 5, which should be different from the first frame, and
  // hence, indicate a corruption.
  scoped_refptr<I420Buffer> different_frame =
      frame_reader->ReadFrame(/*frame_num=*/5);

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kScaleFactor, std::nullopt);
  EXPECT_GT(frame_pair_corruption_score.CalculateScore(
                /*qp=*/1, *frame,
                *GetDowscaledFrame(different_frame, /*downscale_factor=*/0.25)),
            0.5);
}

TEST(FramePairCorruptionScorerTest,
     HalfScaledDifferentFrameResultsInCorruptionLogisticFunction) {
  std::unique_ptr<FrameReader> frame_reader = GetFrameGenerator();
  scoped_refptr<I420Buffer> frame = frame_reader->PullFrame();

  // Get frame number 5, which should be different from the first frame, and
  // hence, indicate a corruption.
  scoped_refptr<I420Buffer> different_frame =
      frame_reader->ReadFrame(/*frame_num=*/5);

  FramePairCorruptionScorer frame_pair_corruption_score(
      kCodecName, kGrowthRate, kMidpoint, std::nullopt);
  EXPECT_GT(frame_pair_corruption_score.CalculateScore(
                /*qp=*/1, *frame,
                *GetDowscaledFrame(different_frame, /*downscale_factor=*/0.25)),
            0.5);
}

}  // namespace
}  // namespace webrtc
