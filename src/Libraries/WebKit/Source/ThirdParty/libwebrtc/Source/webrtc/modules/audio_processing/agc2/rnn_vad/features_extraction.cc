/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#include "modules/audio_processing/agc2/rnn_vad/features_extraction.h"

#include <array>

#include "modules/audio_processing/agc2/rnn_vad/lp_residual.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace rnn_vad {
namespace {

// Computed as `scipy.signal.butter(N=2, Wn=60/24000, btype='highpass')`.
constexpr BiQuadFilter::Config kHpfConfig24k{
    {0.99446179f, -1.98892358f, 0.99446179f},
    {-1.98889291f, 0.98895425f}};

}  // namespace

FeaturesExtractor::FeaturesExtractor(const AvailableCpuFeatures& cpu_features)
    : use_high_pass_filter_(false),
      hpf_(kHpfConfig24k),
      pitch_buf_24kHz_(),
      pitch_buf_24kHz_view_(pitch_buf_24kHz_.GetBufferView()),
      lp_residual_(kBufSize24kHz),
      lp_residual_view_(lp_residual_.data(), kBufSize24kHz),
      pitch_estimator_(cpu_features),
      reference_frame_view_(pitch_buf_24kHz_.GetMostRecentValuesView()) {
  RTC_DCHECK_EQ(kBufSize24kHz, lp_residual_.size());
  Reset();
}

FeaturesExtractor::~FeaturesExtractor() = default;

void FeaturesExtractor::Reset() {
  pitch_buf_24kHz_.Reset();
  spectral_features_extractor_.Reset();
  if (use_high_pass_filter_) {
    hpf_.Reset();
  }
}

bool FeaturesExtractor::CheckSilenceComputeFeatures(
    rtc::ArrayView<const float, kFrameSize10ms24kHz> samples,
    rtc::ArrayView<float, kFeatureVectorSize> feature_vector) {
  // Pre-processing.
  if (use_high_pass_filter_) {
    std::array<float, kFrameSize10ms24kHz> samples_filtered;
    hpf_.Process(samples, samples_filtered);
    // Feed buffer with the pre-processed version of `samples`.
    pitch_buf_24kHz_.Push(samples_filtered);
  } else {
    // Feed buffer with `samples`.
    pitch_buf_24kHz_.Push(samples);
  }
  // Extract the LP residual.
  float lpc_coeffs[kNumLpcCoefficients];
  ComputeAndPostProcessLpcCoefficients(pitch_buf_24kHz_view_, lpc_coeffs);
  ComputeLpResidual(lpc_coeffs, pitch_buf_24kHz_view_, lp_residual_view_);
  // Estimate pitch on the LP-residual and write the normalized pitch period
  // into the output vector (normalization based on training data stats).
  pitch_period_48kHz_ = pitch_estimator_.Estimate(lp_residual_view_);
  feature_vector[kFeatureVectorSize - 2] = 0.01f * (pitch_period_48kHz_ - 300);
  // Extract lagged frames (according to the estimated pitch period).
  RTC_DCHECK_LE(pitch_period_48kHz_ / 2, kMaxPitch24kHz);
  auto lagged_frame = pitch_buf_24kHz_view_.subview(
      kMaxPitch24kHz - pitch_period_48kHz_ / 2, kFrameSize20ms24kHz);
  // Analyze reference and lagged frames checking if silence has been detected
  // and write the feature vector.
  return spectral_features_extractor_.CheckSilenceComputeFeatures(
      reference_frame_view_, {lagged_frame.data(), kFrameSize20ms24kHz},
      {feature_vector.data() + kNumLowerBands, kNumBands - kNumLowerBands},
      {feature_vector.data(), kNumLowerBands},
      {feature_vector.data() + kNumBands, kNumLowerBands},
      {feature_vector.data() + kNumBands + kNumLowerBands, kNumLowerBands},
      {feature_vector.data() + kNumBands + 2 * kNumLowerBands, kNumLowerBands},
      &feature_vector[kFeatureVectorSize - 1]);
}

}  // namespace rnn_vad
}  // namespace webrtc
