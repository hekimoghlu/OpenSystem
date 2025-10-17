/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#include "modules/video_coding/utility/framerate_controller_deprecated.h"

#include <stddef.h>

#include "test/gtest.h"

namespace webrtc {

TEST(FramerateControllerDeprecated, KeepTargetFramerate) {
  const float input_framerate_fps = 20;
  const float target_framerate_fps = 5;
  const float max_abs_framerate_error_fps = target_framerate_fps * 0.1f;
  const size_t input_duration_secs = 3;
  const size_t num_input_frames = input_duration_secs * input_framerate_fps;

  FramerateControllerDeprecated framerate_controller(target_framerate_fps);
  size_t num_dropped_frames = 0;
  for (size_t frame_num = 0; frame_num < num_input_frames; ++frame_num) {
    const uint32_t timestamp_ms =
        static_cast<uint32_t>(1000 * frame_num / input_framerate_fps);
    if (framerate_controller.DropFrame(timestamp_ms)) {
      ++num_dropped_frames;
    } else {
      framerate_controller.AddFrame(timestamp_ms);
    }
  }

  const float output_framerate_fps =
      static_cast<float>(num_input_frames - num_dropped_frames) /
      input_duration_secs;
  EXPECT_NEAR(output_framerate_fps, target_framerate_fps,
              max_abs_framerate_error_fps);
}

TEST(FramerateControllerDeprecated, DoNotDropAnyFramesIfTargerEqualsInput) {
  const float input_framerate_fps = 30;
  const size_t input_duration_secs = 3;
  const size_t num_input_frames = input_duration_secs * input_framerate_fps;

  FramerateControllerDeprecated framerate_controller(input_framerate_fps);
  size_t num_dropped_frames = 0;
  for (size_t frame_num = 0; frame_num < num_input_frames; ++frame_num) {
    const uint32_t timestamp_ms =
        static_cast<uint32_t>(1000 * frame_num / input_framerate_fps);
    if (framerate_controller.DropFrame(timestamp_ms)) {
      ++num_dropped_frames;
    } else {
      framerate_controller.AddFrame(timestamp_ms);
    }
  }

  EXPECT_EQ(num_dropped_frames, 0U);
}

TEST(FramerateControllerDeprecated, DoNotDropFrameWhenTimestampJumpsBackward) {
  FramerateControllerDeprecated framerate_controller(30);
  ASSERT_FALSE(framerate_controller.DropFrame(66));
  framerate_controller.AddFrame(66);
  EXPECT_FALSE(framerate_controller.DropFrame(33));
}

TEST(FramerateControllerDeprecated, DropFrameIfItIsTooCloseToPreviousFrame) {
  FramerateControllerDeprecated framerate_controller(30);
  ASSERT_FALSE(framerate_controller.DropFrame(33));
  framerate_controller.AddFrame(33);
  EXPECT_TRUE(framerate_controller.DropFrame(34));
}

TEST(FramerateControllerDeprecated, FrameDroppingStartsFromSecondInputFrame) {
  const float input_framerate_fps = 23;
  const float target_framerate_fps = 19;
  const uint32_t input_frame_duration_ms =
      static_cast<uint32_t>(1000 / input_framerate_fps);
  FramerateControllerDeprecated framerate_controller(target_framerate_fps);
  ASSERT_FALSE(framerate_controller.DropFrame(1 * input_frame_duration_ms));
  framerate_controller.AddFrame(1 * input_frame_duration_ms);
  EXPECT_TRUE(framerate_controller.DropFrame(2 * input_frame_duration_ms));
}

}  // namespace webrtc
