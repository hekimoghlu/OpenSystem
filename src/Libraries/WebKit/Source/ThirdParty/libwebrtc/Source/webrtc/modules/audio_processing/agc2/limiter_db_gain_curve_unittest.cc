/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#include "modules/audio_processing/agc2/limiter_db_gain_curve.h"

#include "rtc_base/gunit.h"

namespace webrtc {

TEST(FixedDigitalGainController2Limiter, ConstructDestruct) {
  LimiterDbGainCurve l;
}

TEST(FixedDigitalGainController2Limiter, GainCurveShouldBeMonotone) {
  LimiterDbGainCurve l;
  float last_output_level = 0.f;
  bool has_last_output_level = false;
  for (float level = -90.f; level <= l.max_input_level_db(); level += 0.5f) {
    const float current_output_level = l.GetOutputLevelDbfs(level);
    if (!has_last_output_level) {
      last_output_level = current_output_level;
      has_last_output_level = true;
    }
    EXPECT_LE(last_output_level, current_output_level);
    last_output_level = current_output_level;
  }
}

TEST(FixedDigitalGainController2Limiter, GainCurveShouldBeContinuous) {
  LimiterDbGainCurve l;
  float last_output_level = 0.f;
  bool has_last_output_level = false;
  constexpr float kMaxDelta = 0.5f;
  for (float level = -90.f; level <= l.max_input_level_db(); level += 0.5f) {
    const float current_output_level = l.GetOutputLevelDbfs(level);
    if (!has_last_output_level) {
      last_output_level = current_output_level;
      has_last_output_level = true;
    }
    EXPECT_LE(current_output_level, last_output_level + kMaxDelta);
    last_output_level = current_output_level;
  }
}

TEST(FixedDigitalGainController2Limiter, OutputGainShouldBeLessThanFullScale) {
  LimiterDbGainCurve l;
  for (float level = -90.f; level <= l.max_input_level_db(); level += 0.5f) {
    const float current_output_level = l.GetOutputLevelDbfs(level);
    EXPECT_LE(current_output_level, 0.f);
  }
}

}  // namespace webrtc
