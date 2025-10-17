/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
#include "modules/audio_processing/vad/pitch_internal.h"

#include <math.h>

#include "test/gtest.h"

namespace webrtc {

TEST(PitchInternalTest, test) {
  const int kSamplingRateHz = 8000;
  const int kNumInputParameters = 4;
  const int kNumOutputParameters = 3;
  // Inputs
  double log_old_gain = log(0.5);
  double gains[] = {0.6, 0.2, 0.5, 0.4};

  double old_lag = 70;
  double lags[] = {90, 111, 122, 50};

  // Expected outputs
  double expected_log_pitch_gain[] = {-0.541212549898316, -1.45672279045507,
                                      -0.80471895621705};
  double expected_log_old_gain = log(gains[kNumInputParameters - 1]);

  double expected_pitch_lag_hz[] = {92.3076923076923, 70.9010339734121,
                                    93.0232558139535};
  double expected_old_lag = lags[kNumInputParameters - 1];

  double log_pitch_gain[kNumOutputParameters];
  double pitch_lag_hz[kNumInputParameters];

  GetSubframesPitchParameters(kSamplingRateHz, gains, lags, kNumInputParameters,
                              kNumOutputParameters, &log_old_gain, &old_lag,
                              log_pitch_gain, pitch_lag_hz);

  for (int n = 0; n < 3; n++) {
    EXPECT_NEAR(pitch_lag_hz[n], expected_pitch_lag_hz[n], 1e-6);
    EXPECT_NEAR(log_pitch_gain[n], expected_log_pitch_gain[n], 1e-8);
  }
  EXPECT_NEAR(old_lag, expected_old_lag, 1e-6);
  EXPECT_NEAR(log_old_gain, expected_log_old_gain, 1e-8);
}

}  // namespace webrtc
