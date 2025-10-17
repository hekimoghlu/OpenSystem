/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_SPECTRAL_FEATURES_INTERNAL_H_
#define MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_SPECTRAL_FEATURES_INTERNAL_H_

#include <stddef.h>

#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/agc2/rnn_vad/common.h"

namespace webrtc {
namespace rnn_vad {

// At a sample rate of 24 kHz, the last 3 Opus bands are beyond the Nyquist
// frequency. However, band #19 gets the contributions from band #18 because
// of the symmetric triangular filter with peak response at 12 kHz.
constexpr int kOpusBands24kHz = 20;
static_assert(kOpusBands24kHz < kNumBands,
              "The number of bands at 24 kHz must be less than those defined "
              "in the Opus scale at 48 kHz.");

// Number of FFT frequency bins covered by each band in the Opus scale at a
// sample rate of 24 kHz for 20 ms frames.
// Declared here for unit testing.
constexpr std::array<int, kOpusBands24kHz - 1> GetOpusScaleNumBins24kHz20ms() {
  return {4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 24, 24, 32, 48};
}

// TODO(bugs.webrtc.org/10480): Move to a separate file.
// Class to compute band-wise spectral features in the Opus perceptual scale
// for 20 ms frames sampled at 24 kHz. The analysis methods apply triangular
// filters with peak response at the each band boundary.
class SpectralCorrelator {
 public:
  // Ctor.
  SpectralCorrelator();
  SpectralCorrelator(const SpectralCorrelator&) = delete;
  SpectralCorrelator& operator=(const SpectralCorrelator&) = delete;
  ~SpectralCorrelator();

  // Computes the band-wise spectral auto-correlations.
  // `x` must:
  //  - have size equal to `kFrameSize20ms24kHz`;
  //  - be encoded as vectors of interleaved real-complex FFT coefficients
  //    where x[1] = y[1] = 0 (the Nyquist frequency coefficient is omitted).
  void ComputeAutoCorrelation(
      rtc::ArrayView<const float> x,
      rtc::ArrayView<float, kOpusBands24kHz> auto_corr) const;

  // Computes the band-wise spectral cross-correlations.
  // `x` and `y` must:
  //  - have size equal to `kFrameSize20ms24kHz`;
  //  - be encoded as vectors of interleaved real-complex FFT coefficients where
  //    x[1] = y[1] = 0 (the Nyquist frequency coefficient is omitted).
  void ComputeCrossCorrelation(
      rtc::ArrayView<const float> x,
      rtc::ArrayView<const float> y,
      rtc::ArrayView<float, kOpusBands24kHz> cross_corr) const;

 private:
  const std::vector<float> weights_;  // Weights for each Fourier coefficient.
};

// TODO(bugs.webrtc.org/10480): Move to anonymous namespace in
// spectral_features.cc. Given a vector of Opus-bands energy coefficients,
// computes the log magnitude spectrum applying smoothing both over time and
// over frequency. Declared here for unit testing.
void ComputeSmoothedLogMagnitudeSpectrum(
    rtc::ArrayView<const float> bands_energy,
    rtc::ArrayView<float, kNumBands> log_bands_energy);

// TODO(bugs.webrtc.org/10480): Move to anonymous namespace in
// spectral_features.cc. Creates a DCT table for arrays having size equal to
// `kNumBands`. Declared here for unit testing.
std::array<float, kNumBands * kNumBands> ComputeDctTable();

// TODO(bugs.webrtc.org/10480): Move to anonymous namespace in
// spectral_features.cc. Computes DCT for `in` given a pre-computed DCT table.
// In-place computation is not allowed and `out` can be smaller than `in` in
// order to only compute the first DCT coefficients. Declared here for unit
// testing.
void ComputeDct(rtc::ArrayView<const float> in,
                rtc::ArrayView<const float, kNumBands * kNumBands> dct_table,
                rtc::ArrayView<float> out);

}  // namespace rnn_vad
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_SPECTRAL_FEATURES_INTERNAL_H_
