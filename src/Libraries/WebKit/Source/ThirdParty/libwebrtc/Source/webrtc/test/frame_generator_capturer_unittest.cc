/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include "test/frame_generator_capturer.h"

#include "test/create_frame_generator_capturer.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {
namespace test {
namespace {
using ::testing::Eq;
using ::testing::Property;

constexpr int kWidth = 640;
constexpr int kHeight = 360;

class MockVideoSinkInterfaceVideoFrame
    : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  MOCK_METHOD(void, OnFrame, (const VideoFrame& frame), (override));
  MOCK_METHOD(void, OnDiscardedFrame, (), (override));
};
}  // namespace

TEST(FrameGeneratorCapturerTest, CreateFromConfig) {
  GlobalSimulatedTimeController time(Timestamp::Seconds(1000));
  FrameGeneratorCapturerConfig config;
  config.squares_video->width = 300;
  config.squares_video->height = 200;
  config.squares_video->framerate = 20;
  auto capturer = CreateFrameGeneratorCapturer(
      time.GetClock(), *time.GetTaskQueueFactory(), config);
  testing::StrictMock<MockVideoSinkInterfaceVideoFrame> mock_sink;
  capturer->AddOrUpdateSink(&mock_sink, rtc::VideoSinkWants());
  capturer->Start();
  EXPECT_CALL(mock_sink, OnFrame(Property(&VideoFrame::width, Eq(300))))
      .Times(21);
  time.AdvanceTime(TimeDelta::Seconds(1));
}

TEST(FrameGeneratorCapturerTest, OnOutputFormatRequest) {
  GlobalSimulatedTimeController time(Timestamp::Seconds(1000));
  FrameGeneratorCapturerConfig config;
  config.squares_video->width = kWidth;
  config.squares_video->height = kHeight;
  config.squares_video->framerate = 20;
  auto capturer = CreateFrameGeneratorCapturer(
      time.GetClock(), *time.GetTaskQueueFactory(), config);
  testing::StrictMock<MockVideoSinkInterfaceVideoFrame> mock_sink;
  capturer->AddOrUpdateSink(&mock_sink, rtc::VideoSinkWants());
  capturer->OnOutputFormatRequest(kWidth / 2, kHeight / 2, /*max_fps=*/10);
  capturer->Start();
  EXPECT_CALL(mock_sink, OnFrame(Property(&VideoFrame::width, Eq(kWidth / 2))))
      .Times(11);
  time.AdvanceTime(TimeDelta::Seconds(1));
}

TEST(FrameGeneratorCapturerTest, ChangeResolution) {
  GlobalSimulatedTimeController time(Timestamp::Seconds(1000));
  FrameGeneratorCapturerConfig config;
  config.squares_video->width = kWidth;
  config.squares_video->height = kHeight;
  config.squares_video->framerate = 20;
  auto capturer = CreateFrameGeneratorCapturer(
      time.GetClock(), *time.GetTaskQueueFactory(), config);
  EXPECT_TRUE(capturer->GetResolution());
  EXPECT_EQ(kWidth, capturer->GetResolution()->width);
  EXPECT_EQ(kHeight, capturer->GetResolution()->height);
  capturer->Start();
  time.AdvanceTime(TimeDelta::Seconds(1));
  ASSERT_TRUE(capturer->GetResolution());
  EXPECT_EQ(kWidth, capturer->GetResolution()->width);
  EXPECT_EQ(kHeight, capturer->GetResolution()->height);

  capturer->ChangeResolution(kWidth / 2, kHeight / 2);
  time.AdvanceTime(TimeDelta::Seconds(1));
  ASSERT_TRUE(capturer->GetResolution());
  EXPECT_EQ(kWidth / 2, capturer->GetResolution()->width);
  EXPECT_EQ(kHeight / 2, capturer->GetResolution()->height);
}

}  // namespace test
}  // namespace webrtc
