/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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

#include <array>
#include <cstddef>

#include "rtc_base/checks.h"

namespace webrtc {
namespace rnn_vad {

PitchEstimator::PitchEstimator(const AvailableCpuFeatures& cpu_features)
    : cpu_features_(cpu_features),
      y_energy_24kHz_(kRefineNumLags24kHz, 0.f),
      pitch_buffer_12kHz_(kBufSize12kHz),
      auto_correlation_12kHz_(kNumLags12kHz) {}

PitchEstimator::~PitchEstimator() = default;

int PitchEstimator::Estimate(
    rtc::ArrayView<const float, kBufSize24kHz> pitch_buffer) {
  rtc::ArrayView<float, kBufSize12kHz> pitch_buffer_12kHz_view(
      pitch_buffer_12kHz_.data(), kBufSize12kHz);
  RTC_DCHECK_EQ(pitch_buffer_12kHz_.size(), pitch_buffer_12kHz_view.size());
  rtc::ArrayView<float, kNumLags12kHz> auto_correlation_12kHz_view(
      auto_correlation_12kHz_.data(), kNumLags12kHz);
  RTC_DCHECK_EQ(auto_correlation_12kHz_.size(),
                auto_correlation_12kHz_view.size());

  // TODO(bugs.chromium.org/10480): Use `cpu_features_` to estimate pitch.
  // Perform the initial pitch search at 12 kHz.
  Decimate2x(pitch_buffer, pitch_buffer_12kHz_view);
  auto_corr_calculator_.ComputeOnPitchBuffer(pitch_buffer_12kHz_view,
                                             auto_correlation_12kHz_view);
  CandidatePitchPeriods pitch_periods = ComputePitchPeriod12kHz(
      pitch_buffer_12kHz_view, auto_correlation_12kHz_view, cpu_features_);
  // The refinement is done using the pitch buffer that contains 24 kHz samples.
  // Therefore, adapt the inverted lags in `pitch_candidates_inv_lags` from 12
  // to 24 kHz.
  pitch_periods.best *= 2;
  pitch_periods.second_best *= 2;

  // Refine the initial pitch period estimation from 12 kHz to 48 kHz.
  // Pre-compute frame energies at 24 kHz.
  rtc::ArrayView<float, kRefineNumLags24kHz> y_energy_24kHz_view(
      y_energy_24kHz_.data(), kRefineNumLags24kHz);
  RTC_DCHECK_EQ(y_energy_24kHz_.size(), y_energy_24kHz_view.size());
  ComputeSlidingFrameSquareEnergies24kHz(pitch_buffer, y_energy_24kHz_view,
                                         cpu_features_);
  // Estimation at 48 kHz.
  const int pitch_lag_48kHz = ComputePitchPeriod48kHz(
      pitch_buffer, y_energy_24kHz_view, pitch_periods, cpu_features_);
  last_pitch_48kHz_ = ComputeExtendedPitchPeriod48kHz(
      pitch_buffer, y_energy_24kHz_view,
      /*initial_pitch_period_48kHz=*/kMaxPitch48kHz - pitch_lag_48kHz,
      last_pitch_48kHz_, cpu_features_);
  return last_pitch_48kHz_.period;
}

}  // namespace rnn_vad
}  // namespace webrtc
