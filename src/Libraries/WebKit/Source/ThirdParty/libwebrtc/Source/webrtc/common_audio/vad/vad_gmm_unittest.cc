/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
#include "common_audio/vad/vad_unittest.h"
#include "test/gtest.h"

extern "C" {
#include "common_audio/vad/vad_gmm.h"
}

namespace webrtc {
namespace test {

// TODO(bugs.webrtc.org/345674543): Fix/enable.
#if defined(__has_feature) && __has_feature(undefined_behavior_sanitizer)
TEST_F(VadTest, DISABLED_vad_gmm) {
#else
TEST_F(VadTest, vad_gmm) {
#endif
  int16_t delta = 0;
  // Input value at mean.
  EXPECT_EQ(1048576, WebRtcVad_GaussianProbability(0, 0, 128, &delta));
  EXPECT_EQ(0, delta);
  EXPECT_EQ(1048576, WebRtcVad_GaussianProbability(16, 128, 128, &delta));
  EXPECT_EQ(0, delta);
  EXPECT_EQ(1048576, WebRtcVad_GaussianProbability(-16, -128, 128, &delta));
  EXPECT_EQ(0, delta);

  // Largest possible input to give non-zero probability.
  EXPECT_EQ(1024, WebRtcVad_GaussianProbability(59, 0, 128, &delta));
  EXPECT_EQ(7552, delta);
  EXPECT_EQ(1024, WebRtcVad_GaussianProbability(75, 128, 128, &delta));
  EXPECT_EQ(7552, delta);
  EXPECT_EQ(1024, WebRtcVad_GaussianProbability(-75, -128, 128, &delta));
  EXPECT_EQ(-7552, delta);

  // Too large input, should give zero probability.
  EXPECT_EQ(0, WebRtcVad_GaussianProbability(105, 0, 128, &delta));
  EXPECT_EQ(13440, delta);
}
}  // namespace test
}  // namespace webrtc
