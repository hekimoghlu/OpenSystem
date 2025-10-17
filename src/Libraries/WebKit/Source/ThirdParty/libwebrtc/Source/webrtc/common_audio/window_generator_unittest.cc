/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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
#include "common_audio/window_generator.h"

#include <cstring>

#include "test/gtest.h"

namespace webrtc {

TEST(WindowGeneratorTest, KaiserBesselDerived) {
  float window[7];

  memset(window, 0, sizeof(window));

  WindowGenerator::KaiserBesselDerived(0.397856f, 2, window);
  ASSERT_NEAR(window[0], 0.707106f, 1e-6f);
  ASSERT_NEAR(window[1], 0.707106f, 1e-6f);
  ASSERT_NEAR(window[2], 0.0f, 1e-6f);
  ASSERT_NEAR(window[3], 0.0f, 1e-6f);
  ASSERT_NEAR(window[4], 0.0f, 1e-6f);
  ASSERT_NEAR(window[5], 0.0f, 1e-6f);
  ASSERT_NEAR(window[6], 0.0f, 1e-6f);

  WindowGenerator::KaiserBesselDerived(0.397856f, 3, window);
  ASSERT_NEAR(window[0], 0.598066f, 1e-6f);
  ASSERT_NEAR(window[1], 0.922358f, 1e-6f);
  ASSERT_NEAR(window[2], 0.598066f, 1e-6f);
  ASSERT_NEAR(window[3], 0.0f, 1e-6f);
  ASSERT_NEAR(window[4], 0.0f, 1e-6f);
  ASSERT_NEAR(window[5], 0.0f, 1e-6f);
  ASSERT_NEAR(window[6], 0.0f, 1e-6f);

  WindowGenerator::KaiserBesselDerived(0.397856f, 6, window);
  ASSERT_NEAR(window[0], 0.458495038865344f, 1e-6f);
  ASSERT_NEAR(window[1], 0.707106781186548f, 1e-6f);
  ASSERT_NEAR(window[2], 0.888696967101760f, 1e-6f);
  ASSERT_NEAR(window[3], 0.888696967101760f, 1e-6f);
  ASSERT_NEAR(window[4], 0.707106781186548f, 1e-6f);
  ASSERT_NEAR(window[5], 0.458495038865344f, 1e-6f);
  ASSERT_NEAR(window[6], 0.0f, 1e-6f);
}

TEST(WindowGeneratorTest, Hanning) {
  float window[7];

  memset(window, 0, sizeof(window));

  window[0] = -1.0f;
  window[1] = -1.0f;
  WindowGenerator::Hanning(2, window);
  ASSERT_NEAR(window[0], 0.0f, 1e-6f);
  ASSERT_NEAR(window[1], 0.0f, 1e-6f);
  ASSERT_NEAR(window[2], 0.0f, 1e-6f);
  ASSERT_NEAR(window[3], 0.0f, 1e-6f);
  ASSERT_NEAR(window[4], 0.0f, 1e-6f);
  ASSERT_NEAR(window[5], 0.0f, 1e-6f);
  ASSERT_NEAR(window[6], 0.0f, 1e-6f);

  window[0] = -1.0f;
  window[2] = -1.0f;
  WindowGenerator::Hanning(3, window);
  ASSERT_NEAR(window[0], 0.0f, 1e-6f);
  ASSERT_NEAR(window[1], 1.0f, 1e-6f);
  ASSERT_NEAR(window[2], 0.0f, 1e-6f);
  ASSERT_NEAR(window[3], 0.0f, 1e-6f);
  ASSERT_NEAR(window[4], 0.0f, 1e-6f);
  ASSERT_NEAR(window[5], 0.0f, 1e-6f);
  ASSERT_NEAR(window[6], 0.0f, 1e-6f);

  window[0] = -1.0f;
  window[5] = -1.0f;
  WindowGenerator::Hanning(6, window);
  ASSERT_NEAR(window[0], 0.0f, 1e-6f);
  ASSERT_NEAR(window[1], 0.345491f, 1e-6f);
  ASSERT_NEAR(window[2], 0.904508f, 1e-6f);
  ASSERT_NEAR(window[3], 0.904508f, 1e-6f);
  ASSERT_NEAR(window[4], 0.345491f, 1e-6f);
  ASSERT_NEAR(window[5], 0.0f, 1e-6f);
  ASSERT_NEAR(window[6], 0.0f, 1e-6f);
}

}  // namespace webrtc
