/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#include "modules/audio_processing/agc2/rnn_vad/auto_correlation.h"

#include "modules/audio_processing/agc2/rnn_vad/common.h"
#include "modules/audio_processing/agc2/rnn_vad/pitch_search_internal.h"
#include "modules/audio_processing/agc2/rnn_vad/test_utils.h"
#include "test/gtest.h"

namespace webrtc {
namespace rnn_vad {
namespace {

// Checks that the auto correlation function produces output within tolerance
// given test input data.
TEST(RnnVadTest, PitchBufferAutoCorrelationWithinTolerance) {
  PitchTestData test_data;
  std::array<float, kBufSize12kHz> pitch_buf_decimated;
  Decimate2x(test_data.PitchBuffer24kHzView(), pitch_buf_decimated);
  std::array<float, kNumLags12kHz> computed_output;
  {
    // TODO(bugs.webrtc.org/8948): Add when the issue is fixed.
    // FloatingPointExceptionObserver fpe_observer;
    AutoCorrelationCalculator auto_corr_calculator;
    auto_corr_calculator.ComputeOnPitchBuffer(pitch_buf_decimated,
                                              computed_output);
  }
  auto auto_corr_view = test_data.AutoCorrelation12kHzView();
  ExpectNearAbsolute({auto_corr_view.data(), auto_corr_view.size()},
                     computed_output, 3e-3f);
}

// Checks that the auto correlation function computes the right thing for a
// simple use case.
TEST(RnnVadTest, CheckAutoCorrelationOnConstantPitchBuffer) {
  // Create constant signal with no pitch.
  std::array<float, kBufSize12kHz> pitch_buf_decimated;
  std::fill(pitch_buf_decimated.begin(), pitch_buf_decimated.end(), 1.f);
  std::array<float, kNumLags12kHz> computed_output;
  {
    // TODO(bugs.webrtc.org/8948): Add when the issue is fixed.
    // FloatingPointExceptionObserver fpe_observer;
    AutoCorrelationCalculator auto_corr_calculator;
    auto_corr_calculator.ComputeOnPitchBuffer(pitch_buf_decimated,
                                              computed_output);
  }
  // The expected output is a vector filled with the same expected
  // auto-correlation value. The latter equals the length of a 20 ms frame.
  constexpr int kFrameSize20ms12kHz = kFrameSize20ms24kHz / 2;
  std::array<float, kNumLags12kHz> expected_output;
  std::fill(expected_output.begin(), expected_output.end(),
            static_cast<float>(kFrameSize20ms12kHz));
  ExpectNearAbsolute(expected_output, computed_output, 4e-5f);
}

}  // namespace
}  // namespace rnn_vad
}  // namespace webrtc
