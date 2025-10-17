/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#include "modules/audio_mixer/audio_frame_manipulator.h"

#include <algorithm>

#include "test/gtest.h"

namespace webrtc {
namespace {

void FillFrameWithConstants(size_t samples_per_channel,
                            size_t number_of_channels,
                            int16_t value,
                            AudioFrame* frame) {
  frame->num_channels_ = number_of_channels;
  frame->samples_per_channel_ = samples_per_channel;
  int16_t* frame_data = frame->mutable_data();
  std::fill(frame_data, frame_data + samples_per_channel * number_of_channels,
            value);
}
}  // namespace

TEST(AudioFrameManipulator, CompareForwardRampWithExpectedResultStereo) {
  constexpr int kSamplesPerChannel = 5;
  constexpr int kNumberOfChannels = 2;

  // Create a frame with values 5, 5, 5, ... and channels & samples as above.
  AudioFrame frame;
  FillFrameWithConstants(kSamplesPerChannel, kNumberOfChannels, 5, &frame);

  Ramp(0.0f, 1.0f, &frame);

  const int total_samples = kSamplesPerChannel * kNumberOfChannels;
  const int16_t expected_result[total_samples] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
  const int16_t* frame_data = frame.data();
  EXPECT_TRUE(
      std::equal(frame_data, frame_data + total_samples, expected_result));
}

TEST(AudioFrameManipulator, CompareBackwardRampWithExpectedResultMono) {
  constexpr int kSamplesPerChannel = 5;
  constexpr int kNumberOfChannels = 1;

  // Create a frame with values 5, 5, 5, ... and channels & samples as above.
  AudioFrame frame;
  FillFrameWithConstants(kSamplesPerChannel, kNumberOfChannels, 5, &frame);

  Ramp(1.0f, 0.0f, &frame);

  const int total_samples = kSamplesPerChannel * kNumberOfChannels;
  const int16_t expected_result[total_samples] = {5, 4, 3, 2, 1};
  const int16_t* frame_data = frame.data();
  EXPECT_TRUE(
      std::equal(frame_data, frame_data + total_samples, expected_result));
}

}  // namespace webrtc
