/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
#include "modules/audio_processing/agc2/rnn_vad/pitch_search.h"

#include <algorithm>
#include <vector>

#include "modules/audio_processing/agc2/cpu_features.h"
#include "modules/audio_processing/agc2/rnn_vad/pitch_search_internal.h"
#include "modules/audio_processing/agc2/rnn_vad/test_utils.h"
// TODO(bugs.webrtc.org/8948): Add when the issue is fixed.
// #include "test/fpe_observer.h"
#include "test/gtest.h"

namespace webrtc {
namespace rnn_vad {

// Checks that the computed pitch period is bit-exact and that the computed
// pitch gain is within tolerance given test input data.
TEST(RnnVadTest, PitchSearchWithinTolerance) {
  ChunksFileReader reader = CreateLpResidualAndPitchInfoReader();
  const int num_frames = std::min(reader.num_chunks, 300);  // Max 3 s.
  std::vector<float> lp_residual(kBufSize24kHz);
  float expected_pitch_period, expected_pitch_strength;
  const AvailableCpuFeatures cpu_features = GetAvailableCpuFeatures();
  PitchEstimator pitch_estimator(cpu_features);
  {
    // TODO(bugs.webrtc.org/8948): Add when the issue is fixed.
    // FloatingPointExceptionObserver fpe_observer;
    for (int i = 0; i < num_frames; ++i) {
      SCOPED_TRACE(i);
      ASSERT_TRUE(reader.reader->ReadChunk(lp_residual));
      ASSERT_TRUE(reader.reader->ReadValue(expected_pitch_period));
      ASSERT_TRUE(reader.reader->ReadValue(expected_pitch_strength));
      int pitch_period =
          pitch_estimator.Estimate({lp_residual.data(), kBufSize24kHz});
      EXPECT_EQ(expected_pitch_period, pitch_period);
      EXPECT_NEAR(expected_pitch_strength,
                  pitch_estimator.GetLastPitchStrengthForTesting(), 15e-6f);
    }
  }
}

}  // namespace rnn_vad
}  // namespace webrtc
