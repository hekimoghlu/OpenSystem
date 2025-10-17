/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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
// MSVC++ requires this to be set before any other includes to get M_PI.
#define _USE_MATH_DEFINES

#include "common_audio/resampler/sinusoidal_linear_chirp_source.h"

#include <math.h>

namespace webrtc {

SinusoidalLinearChirpSource::SinusoidalLinearChirpSource(int sample_rate,
                                                         size_t samples,
                                                         double max_frequency,
                                                         double delay_samples)
    : sample_rate_(sample_rate),
      total_samples_(samples),
      max_frequency_(max_frequency),
      current_index_(0),
      delay_samples_(delay_samples) {
  // Chirp rate.
  double duration = static_cast<double>(total_samples_) / sample_rate_;
  k_ = (max_frequency_ - kMinFrequency) / duration;
}

void SinusoidalLinearChirpSource::Run(size_t frames, float* destination) {
  for (size_t i = 0; i < frames; ++i, ++current_index_) {
    // Filter out frequencies higher than Nyquist.
    if (Frequency(current_index_) > 0.5 * sample_rate_) {
      destination[i] = 0;
    } else {
      // Calculate time in seconds.
      if (current_index_ < delay_samples_) {
        destination[i] = 0;
      } else {
        // Sinusoidal linear chirp.
        double t = (current_index_ - delay_samples_) / sample_rate_;
        destination[i] = sin(2 * M_PI * (kMinFrequency * t + (k_ / 2) * t * t));
      }
    }
  }
}

double SinusoidalLinearChirpSource::Frequency(size_t position) {
  return kMinFrequency + (position - delay_samples_) *
                             (max_frequency_ - kMinFrequency) / total_samples_;
}

}  // namespace webrtc
