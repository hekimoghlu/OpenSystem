/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#ifndef PC_TEST_FRAME_GENERATOR_CAPTURER_VIDEO_TRACK_SOURCE_H_
#define PC_TEST_FRAME_GENERATOR_CAPTURER_VIDEO_TRACK_SOURCE_H_

#include <memory>
#include <utility>

#include "api/task_queue/default_task_queue_factory.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/test/create_frame_generator.h"
#include "pc/video_track_source.h"
#include "test/frame_generator_capturer.h"

namespace webrtc {

// Implements a VideoTrackSourceInterface to be used for creating VideoTracks.
// The video source is generated using a FrameGeneratorCapturer, specifically
// a SquareGenerator that generates frames with randomly sized and colored
// squares.
class FrameGeneratorCapturerVideoTrackSource : public VideoTrackSource {
 public:
  static const int kDefaultFramesPerSecond = 30;
  static const int kDefaultWidth = 640;
  static const int kDefaultHeight = 480;
  static const int kNumSquaresGenerated = 50;

  struct Config {
    int frames_per_second = kDefaultFramesPerSecond;
    int width = kDefaultWidth;
    int height = kDefaultHeight;
    int num_squares_generated = 50;
  };

  FrameGeneratorCapturerVideoTrackSource(Config config,
                                         Clock* clock,
                                         bool is_screencast)
      : VideoTrackSource(false /* remote */),
        task_queue_factory_(CreateDefaultTaskQueueFactory()),
        is_screencast_(is_screencast) {
    video_capturer_ = std::make_unique<test::FrameGeneratorCapturer>(
        clock,
        test::CreateSquareFrameGenerator(config.width, config.height,
                                         std::nullopt,
                                         config.num_squares_generated),
        config.frames_per_second, *task_queue_factory_);
    video_capturer_->Init();
  }

  FrameGeneratorCapturerVideoTrackSource(
      std::unique_ptr<test::FrameGeneratorCapturer> video_capturer,
      bool is_screencast)
      : VideoTrackSource(false /* remote */),
        video_capturer_(std::move(video_capturer)),
        is_screencast_(is_screencast) {}

  ~FrameGeneratorCapturerVideoTrackSource() = default;

  void Start() {
    SetState(kLive);
    video_capturer_->Start();
  }

  void Stop() {
    SetState(kMuted);
    video_capturer_->Stop();
  }

  bool is_screencast() const override { return is_screencast_; }

 protected:
  rtc::VideoSourceInterface<VideoFrame>* source() override {
    return video_capturer_.get();
  }

 private:
  const std::unique_ptr<TaskQueueFactory> task_queue_factory_;
  std::unique_ptr<test::FrameGeneratorCapturer> video_capturer_;
  const bool is_screencast_;
};

}  // namespace webrtc

#endif  // PC_TEST_FRAME_GENERATOR_CAPTURER_VIDEO_TRACK_SOURCE_H_
