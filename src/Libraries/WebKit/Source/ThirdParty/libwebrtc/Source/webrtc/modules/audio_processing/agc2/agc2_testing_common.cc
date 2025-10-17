/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include "modules/audio_processing/agc2/agc2_testing_common.h"

#include <math.h>

#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

std::vector<double> LinSpace(double l, double r, int num_points) {
  RTC_CHECK_GE(num_points, 2);
  std::vector<double> points(num_points);
  const double step = (r - l) / (num_points - 1.0);
  points[0] = l;
  for (int i = 1; i < num_points - 1; i++) {
    points[i] = static_cast<double>(l) + i * step;
  }
  points[num_points - 1] = r;
  return points;
}

WhiteNoiseGenerator::WhiteNoiseGenerator(int min_amplitude, int max_amplitude)
    : rand_gen_(42),
      min_amplitude_(min_amplitude),
      max_amplitude_(max_amplitude) {
  RTC_DCHECK_LT(min_amplitude_, max_amplitude_);
  RTC_DCHECK_LE(kMinS16, min_amplitude_);
  RTC_DCHECK_LE(min_amplitude_, kMaxS16);
  RTC_DCHECK_LE(kMinS16, max_amplitude_);
  RTC_DCHECK_LE(max_amplitude_, kMaxS16);
}

float WhiteNoiseGenerator::operator()() {
  return static_cast<float>(rand_gen_.Rand(min_amplitude_, max_amplitude_));
}

SineGenerator::SineGenerator(float amplitude,
                             float frequency_hz,
                             int sample_rate_hz)
    : amplitude_(amplitude),
      frequency_hz_(frequency_hz),
      sample_rate_hz_(sample_rate_hz),
      x_radians_(0.0f) {
  RTC_DCHECK_GT(amplitude_, 0);
  RTC_DCHECK_LE(amplitude_, kMaxS16);
}

float SineGenerator::operator()() {
  constexpr float kPi = 3.1415926536f;
  x_radians_ += frequency_hz_ / sample_rate_hz_ * 2 * kPi;
  if (x_radians_ >= 2 * kPi) {
    x_radians_ -= 2 * kPi;
  }
  // Use sinf instead of std::sinf for libstdc++ compatibility.
  return amplitude_ * sinf(x_radians_);
}

PulseGenerator::PulseGenerator(float pulse_amplitude,
                               float no_pulse_amplitude,
                               float frequency_hz,
                               int sample_rate_hz)
    : pulse_amplitude_(pulse_amplitude),
      no_pulse_amplitude_(no_pulse_amplitude),
      samples_period_(
          static_cast<int>(static_cast<float>(sample_rate_hz) / frequency_hz)),
      sample_counter_(0) {
  RTC_DCHECK_GE(pulse_amplitude_, kMinS16);
  RTC_DCHECK_LE(pulse_amplitude_, kMaxS16);
  RTC_DCHECK_GT(no_pulse_amplitude_, kMinS16);
  RTC_DCHECK_LE(no_pulse_amplitude_, kMaxS16);
  RTC_DCHECK_GT(sample_rate_hz, frequency_hz);
}

float PulseGenerator::operator()() {
  sample_counter_++;
  if (sample_counter_ >= samples_period_) {
    sample_counter_ -= samples_period_;
  }
  return static_cast<float>(sample_counter_ == 0 ? pulse_amplitude_
                                                 : no_pulse_amplitude_);
}

}  // namespace test
}  // namespace webrtc
