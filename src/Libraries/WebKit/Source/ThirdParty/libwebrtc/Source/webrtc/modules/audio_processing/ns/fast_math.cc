/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
#include "modules/audio_processing/ns/fast_math.h"

#include <math.h>
#include <stdint.h>

#include "rtc_base/checks.h"

namespace webrtc {

namespace {

float FastLog2f(float in) {
  RTC_DCHECK_GT(in, .0f);
  // Read and interpret float as uint32_t and then cast to float.
  // This is done to extract the exponent (bits 30 - 23).
  // "Right shift" of the exponent is then performed by multiplying
  // with the constant (1/2^23). Finally, we subtract a constant to
  // remove the bias (https://en.wikipedia.org/wiki/Exponent_bias).
  union {
    float dummy;
    uint32_t a;
  } x = {in};
  float out = x.a;
  out *= 1.1920929e-7f;  // 1/2^23
  out -= 126.942695f;    // Remove bias.
  return out;
}

}  // namespace

float SqrtFastApproximation(float f) {
  // TODO(peah): Add fast approximate implementation.
  return sqrtf(f);
}

float Pow2Approximation(float p) {
  // TODO(peah): Add fast approximate implementation.
  return powf(2.f, p);
}

float PowApproximation(float x, float p) {
  return Pow2Approximation(p * FastLog2f(x));
}

float LogApproximation(float x) {
  constexpr float kLogOf2 = 0.69314718056f;
  return FastLog2f(x) * kLogOf2;
}

void LogApproximation(rtc::ArrayView<const float> x, rtc::ArrayView<float> y) {
  for (size_t k = 0; k < x.size(); ++k) {
    y[k] = LogApproximation(x[k]);
  }
}

float ExpApproximation(float x) {
  constexpr float kLog10Ofe = 0.4342944819f;
  return PowApproximation(10.f, x * kLog10Ofe);
}

void ExpApproximation(rtc::ArrayView<const float> x, rtc::ArrayView<float> y) {
  for (size_t k = 0; k < x.size(); ++k) {
    y[k] = ExpApproximation(x[k]);
  }
}

void ExpApproximationSignFlip(rtc::ArrayView<const float> x,
                              rtc::ArrayView<float> y) {
  for (size_t k = 0; k < x.size(); ++k) {
    y[k] = ExpApproximation(-x[k]);
  }
}

}  // namespace webrtc
