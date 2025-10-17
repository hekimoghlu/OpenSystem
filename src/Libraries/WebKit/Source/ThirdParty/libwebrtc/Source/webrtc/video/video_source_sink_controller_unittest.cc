/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
#include "video/video_source_sink_controller.h"

#include <limits>

#include "api/video/video_frame.h"
#include "api/video/video_source_interface.h"
#include "call/adaptation/video_source_restrictions.h"
#include "test/gmock.h"
#include "test/gtest.h"

using testing::_;

namespace webrtc {

namespace {

using FrameSize = rtc::VideoSinkWants::FrameSize;
constexpr int kIntUnconstrained = std::numeric_limits<int>::max();

class MockVideoSinkWithVideoFrame : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  ~MockVideoSinkWithVideoFrame() override {}

  MOCK_METHOD(void, OnFrame, (const VideoFrame& frame), (override));
  MOCK_METHOD(void, OnDiscardedFrame, (), (override));
};

class MockVideoSourceWithVideoFrame
    : public rtc::VideoSourceInterface<VideoFrame> {
 public:
  ~MockVideoSourceWithVideoFrame() override {}

  MOCK_METHOD(void,
              AddOrUpdateSink,
              (rtc::VideoSinkInterface<VideoFrame>*,
               const rtc::VideoSinkWants&),
              (override));
  MOCK_METHOD(void,
              RemoveSink,
              (rtc::VideoSinkInterface<VideoFrame>*),
              (override));
  MOCK_METHOD(void, RequestRefreshFrame, (), (override));
};

}  // namespace

TEST(VideoSourceSinkControllerTest, UnconstrainedByDefault) {
  MockVideoSinkWithVideoFrame sink;
  MockVideoSourceWithVideoFrame source;
  VideoSourceSinkController controller(&sink, &source);
  EXPECT_EQ(controller.restrictions(), VideoSourceRestrictions());
  EXPECT_FALSE(controller.pixels_per_frame_upper_limit().has_value());
  EXPECT_FALSE(controller.frame_rate_upper_limit().has_value());
  EXPECT_FALSE(controller.rotation_applied());
  EXPECT_FALSE(controller.scale_resolution_down_to().has_value());
  EXPECT_EQ(controller.resolution_alignment(), 1);

  EXPECT_CALL(source, AddOrUpdateSink(_, _))
      .WillOnce([](rtc::VideoSinkInterface<VideoFrame>* sink,
                   const rtc::VideoSinkWants& wants) {
        EXPECT_FALSE(wants.rotation_applied);
        EXPECT_EQ(wants.max_pixel_count, kIntUnconstrained);
        EXPECT_EQ(wants.target_pixel_count, std::nullopt);
        EXPECT_EQ(wants.max_framerate_fps, kIntUnconstrained);
        EXPECT_EQ(wants.resolution_alignment, 1);
        EXPECT_FALSE(wants.requested_resolution.has_value());
      });
  controller.PushSourceSinkSettings();
}

TEST(VideoSourceSinkControllerTest, VideoRestrictionsToSinkWants) {
  MockVideoSinkWithVideoFrame sink;
  MockVideoSourceWithVideoFrame source;
  VideoSourceSinkController controller(&sink, &source);

  VideoSourceRestrictions restrictions = controller.restrictions();
  // max_pixels_per_frame() maps to `max_pixel_count`.
  restrictions.set_max_pixels_per_frame(42u);
  // target_pixels_per_frame() maps to `target_pixel_count`.
  restrictions.set_target_pixels_per_frame(200u);
  // max_frame_rate() maps to `max_framerate_fps`.
  restrictions.set_max_frame_rate(30.0);
  controller.SetRestrictions(restrictions);
  EXPECT_CALL(source, AddOrUpdateSink(_, _))
      .WillOnce([](rtc::VideoSinkInterface<VideoFrame>* sink,
                   const rtc::VideoSinkWants& wants) {
        EXPECT_EQ(wants.max_pixel_count, 42);
        EXPECT_EQ(wants.target_pixel_count, 200);
        EXPECT_EQ(wants.max_framerate_fps, 30);
      });
  controller.PushSourceSinkSettings();

  // pixels_per_frame_upper_limit() caps `max_pixel_count`.
  controller.SetPixelsPerFrameUpperLimit(24);
  // frame_rate_upper_limit() caps `max_framerate_fps`.
  controller.SetFrameRateUpperLimit(10.0);

  EXPECT_CALL(source, AddOrUpdateSink(_, _))
      .WillOnce([](rtc::VideoSinkInterface<VideoFrame>* sink,
                   const rtc::VideoSinkWants& wants) {
        EXPECT_EQ(wants.max_pixel_count, 24);
        EXPECT_EQ(wants.max_framerate_fps, 10);
      });
  controller.PushSourceSinkSettings();
}

TEST(VideoSourceSinkControllerTest, RotationApplied) {
  MockVideoSinkWithVideoFrame sink;
  MockVideoSourceWithVideoFrame source;
  VideoSourceSinkController controller(&sink, &source);
  controller.SetRotationApplied(true);
  EXPECT_TRUE(controller.rotation_applied());

  EXPECT_CALL(source, AddOrUpdateSink(_, _))
      .WillOnce([](rtc::VideoSinkInterface<VideoFrame>* sink,
                   const rtc::VideoSinkWants& wants) {
        EXPECT_TRUE(wants.rotation_applied);
      });
  controller.PushSourceSinkSettings();
}

TEST(VideoSourceSinkControllerTest, ResolutionAlignment) {
  MockVideoSinkWithVideoFrame sink;
  MockVideoSourceWithVideoFrame source;
  VideoSourceSinkController controller(&sink, &source);
  controller.SetResolutionAlignment(13);
  EXPECT_EQ(controller.resolution_alignment(), 13);

  EXPECT_CALL(source, AddOrUpdateSink(_, _))
      .WillOnce([](rtc::VideoSinkInterface<VideoFrame>* sink,
                   const rtc::VideoSinkWants& wants) {
        EXPECT_EQ(wants.resolution_alignment, 13);
      });
  controller.PushSourceSinkSettings();
}

TEST(VideoSourceSinkControllerTest,
     PushSourceSinkSettingsWithoutSourceDoesNotCrash) {
  MockVideoSinkWithVideoFrame sink;
  VideoSourceSinkController controller(&sink, nullptr);
  controller.PushSourceSinkSettings();
}

TEST(VideoSourceSinkControllerTest, RequestsRefreshFrameWithSource) {
  MockVideoSinkWithVideoFrame sink;
  MockVideoSourceWithVideoFrame source;
  VideoSourceSinkController controller(&sink, &source);
  EXPECT_CALL(source, RequestRefreshFrame);
  controller.RequestRefreshFrame();
}

TEST(VideoSourceSinkControllerTest,
     RequestsRefreshFrameWithoutSourceDoesNotCrash) {
  MockVideoSinkWithVideoFrame sink;
  VideoSourceSinkController controller(&sink, nullptr);
  controller.RequestRefreshFrame();
}

TEST(VideoSourceSinkControllerTest, ScaleResolutionDownToPropagatesToWants) {
  MockVideoSinkWithVideoFrame sink;
  MockVideoSourceWithVideoFrame source;
  VideoSourceSinkController controller(&sink, &source);
  controller.SetScaleResolutionDownTo(FrameSize(640, 360));
  EXPECT_TRUE(controller.scale_resolution_down_to().has_value());

  EXPECT_CALL(source, AddOrUpdateSink(_, _))
      .WillOnce([](rtc::VideoSinkInterface<VideoFrame>* sink,
                   const rtc::VideoSinkWants& wants) {
        EXPECT_EQ(*wants.requested_resolution, FrameSize(640, 360));
      });
  controller.PushSourceSinkSettings();
}

TEST(VideoSourceSinkControllerTest, ActivePropagatesToWants) {
  MockVideoSinkWithVideoFrame sink;
  MockVideoSourceWithVideoFrame source;
  VideoSourceSinkController controller(&sink, &source);
  controller.SetActive(true);
  EXPECT_TRUE(controller.active());

  EXPECT_CALL(source, AddOrUpdateSink(_, _))
      .WillOnce([](rtc::VideoSinkInterface<VideoFrame>* sink,
                   const rtc::VideoSinkWants& wants) {
        EXPECT_TRUE(wants.is_active);
      });
  controller.PushSourceSinkSettings();
}

}  // namespace webrtc
