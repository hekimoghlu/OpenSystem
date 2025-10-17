/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
// Modified from the Chromium original here:
// src/media/base/sinc_resampler_unittest.cc

#ifndef COMMON_AUDIO_RESAMPLER_SINUSOIDAL_LINEAR_CHIRP_SOURCE_H_
#define COMMON_AUDIO_RESAMPLER_SINUSOIDAL_LINEAR_CHIRP_SOURCE_H_

#include "common_audio/resampler/sinc_resampler.h"

namespace webrtc {

// Fake audio source for testing the resampler.  Generates a sinusoidal linear
// chirp (http://en.wikipedia.org/wiki/Chirp) which can be tuned to stress the
// resampler for the specific sample rate conversion being used.
class SinusoidalLinearChirpSource : public SincResamplerCallback {
 public:
  // `delay_samples` can be used to insert a fractional sample delay into the
  // source.  It will produce zeros until non-negative time is reached.
  SinusoidalLinearChirpSource(int sample_rate,
                              size_t samples,
                              double max_frequency,
                              double delay_samples);

  ~SinusoidalLinearChirpSource() override {}

  SinusoidalLinearChirpSource(const SinusoidalLinearChirpSource&) = delete;
  SinusoidalLinearChirpSource& operator=(const SinusoidalLinearChirpSource&) =
      delete;

  void Run(size_t frames, float* destination) override;

  double Frequency(size_t position);

 private:
  static constexpr int kMinFrequency = 5;

  int sample_rate_;
  size_t total_samples_;
  double max_frequency_;
  double k_;
  size_t current_index_;
  double delay_samples_;
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_RESAMPLER_SINUSOIDAL_LINEAR_CHIRP_SOURCE_H_
