/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#define _USE_MATH_DEFINES

#include "common_audio/window_generator.h"

#include <cmath>
#include <complex>

#include "rtc_base/checks.h"

using std::complex;

namespace {

// Modified Bessel function of order 0 for complex inputs.
complex<float> I0(complex<float> x) {
  complex<float> y = x / 3.75f;
  y *= y;
  return 1.0f + y * (3.5156229f +
                     y * (3.0899424f +
                          y * (1.2067492f +
                               y * (0.2659732f +
                                    y * (0.360768e-1f + y * 0.45813e-2f)))));
}

}  // namespace

namespace webrtc {

void WindowGenerator::Hanning(int length, float* window) {
  RTC_CHECK_GT(length, 1);
  RTC_CHECK(window != nullptr);
  for (int i = 0; i < length; ++i) {
    window[i] =
        0.5f * (1 - cosf(2 * static_cast<float>(M_PI) * i / (length - 1)));
  }
}

void WindowGenerator::KaiserBesselDerived(float alpha,
                                          size_t length,
                                          float* window) {
  RTC_CHECK_GT(length, 1U);
  RTC_CHECK(window != nullptr);

  const size_t half = (length + 1) / 2;
  float sum = 0.0f;

  for (size_t i = 0; i <= half; ++i) {
    complex<float> r = (4.0f * i) / length - 1.0f;
    sum += I0(static_cast<float>(M_PI) * alpha * sqrt(1.0f - r * r)).real();
    window[i] = sum;
  }
  for (size_t i = length - 1; i >= half; --i) {
    window[length - i - 1] = sqrtf(window[length - i - 1] / sum);
    window[i] = window[length - i - 1];
  }
  if (length % 2 == 1) {
    window[half - 1] = sqrtf(window[half - 1] / sum);
  }
}

}  // namespace webrtc
