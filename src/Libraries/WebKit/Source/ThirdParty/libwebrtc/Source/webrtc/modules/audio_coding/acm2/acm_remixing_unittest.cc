/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#include "modules/audio_coding/acm2/acm_remixing.h"

#include <vector>

#include "api/audio/audio_frame.h"
#include "system_wrappers/include/clock.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

using ::testing::AllOf;
using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::SizeIs;

namespace webrtc {

TEST(AcmRemixing, DownMixFrame) {
  std::vector<int16_t> out(480, 0);
  AudioFrame in;
  InterleavedView<int16_t> const in_data = in.mutable_data(480, 2);
  for (size_t k = 0; k < in_data.samples_per_channel(); ++k) {
    in_data[2 * k] = 2;
    in_data[2 * k + 1] = 0;
  }

  DownMixFrame(in, out);

  EXPECT_THAT(out, AllOf(SizeIs(480), Each(1)));
}

TEST(AcmRemixing, DownMixMutedFrame) {
  std::vector<int16_t> out(480, 0);
  AudioFrame in;
  in.num_channels_ = 2;
  in.samples_per_channel_ = 480;

  int16_t* const in_data = in.mutable_data();
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    in_data[2 * k] = 2;
    in_data[2 * k + 1] = 0;
  }

  in.Mute();

  DownMixFrame(in, out);

  EXPECT_THAT(out, AllOf(SizeIs(480), Each(0)));
}

TEST(AcmRemixing, RemixMutedStereoFrameTo6Channels) {
  std::vector<int16_t> out(480, 0);
  AudioFrame in;
  in.num_channels_ = 2;
  in.samples_per_channel_ = 480;

  int16_t* const in_data = in.mutable_data();
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    in_data[2 * k] = 1;
    in_data[2 * k + 1] = 2;
  }
  in.Mute();

  ReMixFrame(in, 6, &out);
  EXPECT_EQ(6 * 480u, out.size());

  EXPECT_THAT(out, AllOf(SizeIs(in.samples_per_channel_ * 6), Each(0)));
}

TEST(AcmRemixing, RemixStereoFrameTo6Channels) {
  std::vector<int16_t> out(480, 0);
  AudioFrame in;
  in.num_channels_ = 2;
  in.samples_per_channel_ = 480;

  int16_t* const in_data = in.mutable_data();
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    in_data[2 * k] = 1;
    in_data[2 * k + 1] = 2;
  }

  ReMixFrame(in, 6, &out);
  EXPECT_EQ(6 * 480u, out.size());

  std::vector<int16_t> expected_output(in.samples_per_channel_ * 6);
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    expected_output[6 * k] = 1;
    expected_output[6 * k + 1] = 2;
  }

  EXPECT_THAT(out, ElementsAreArray(expected_output));
}

TEST(AcmRemixing, RemixMonoFrameTo6Channels) {
  std::vector<int16_t> out(480, 0);
  AudioFrame in;
  in.num_channels_ = 1;
  in.samples_per_channel_ = 480;

  int16_t* const in_data = in.mutable_data();
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    in_data[k] = 1;
  }

  ReMixFrame(in, 6, &out);
  EXPECT_EQ(6 * 480u, out.size());

  std::vector<int16_t> expected_output(in.samples_per_channel_ * 6, 0);
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    expected_output[6 * k] = 1;
    expected_output[6 * k + 1] = 1;
  }

  EXPECT_THAT(out, ElementsAreArray(expected_output));
}

TEST(AcmRemixing, RemixStereoFrameToMono) {
  std::vector<int16_t> out(480, 0);
  AudioFrame in;
  in.num_channels_ = 2;
  in.samples_per_channel_ = 480;

  int16_t* const in_data = in.mutable_data();
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    in_data[2 * k] = 2;
    in_data[2 * k + 1] = 0;
  }

  ReMixFrame(in, 1, &out);
  EXPECT_EQ(480u, out.size());

  EXPECT_THAT(out, AllOf(SizeIs(in.samples_per_channel_), Each(1)));
}

TEST(AcmRemixing, RemixMonoFrameToStereo) {
  std::vector<int16_t> out(480, 0);
  AudioFrame in;
  in.num_channels_ = 1;
  in.samples_per_channel_ = 480;

  int16_t* const in_data = in.mutable_data();
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    in_data[k] = 1;
  }

  ReMixFrame(in, 2, &out);
  EXPECT_EQ(960u, out.size());

  EXPECT_THAT(out, AllOf(SizeIs(2 * in.samples_per_channel_), Each(1)));
}

TEST(AcmRemixing, Remix3ChannelFrameToStereo) {
  std::vector<int16_t> out(480, 0);
  AudioFrame in;
  in.num_channels_ = 3;
  in.samples_per_channel_ = 480;

  int16_t* const in_data = in.mutable_data();
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    for (size_t j = 0; j < 3; ++j) {
      in_data[3 * k + j] = j;
    }
  }

  ReMixFrame(in, 2, &out);
  EXPECT_EQ(2 * 480u, out.size());

  std::vector<int16_t> expected_output(in.samples_per_channel_ * 2);
  for (size_t k = 0; k < in.samples_per_channel_; ++k) {
    for (size_t j = 0; j < 2; ++j) {
      expected_output[2 * k + j] = static_cast<int>(j);
    }
  }

  EXPECT_THAT(out, ElementsAreArray(expected_output));
}

}  // namespace webrtc
