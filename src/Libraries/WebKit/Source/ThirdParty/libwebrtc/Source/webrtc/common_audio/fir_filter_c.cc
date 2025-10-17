/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#include "common_audio/fir_filter_c.h"

#include <string.h>

#include <memory>

#include "rtc_base/checks.h"

namespace webrtc {

FIRFilterC::~FIRFilterC() {}

FIRFilterC::FIRFilterC(const float* coefficients, size_t coefficients_length)
    : coefficients_length_(coefficients_length),
      state_length_(coefficients_length - 1),
      coefficients_(new float[coefficients_length_]),
      state_(new float[state_length_]) {
  for (size_t i = 0; i < coefficients_length_; ++i) {
    coefficients_[i] = coefficients[coefficients_length_ - i - 1];
  }
  memset(state_.get(), 0, state_length_ * sizeof(state_[0]));
}

void FIRFilterC::Filter(const float* in, size_t length, float* out) {
  RTC_DCHECK_GT(length, 0);

  // Convolves the input signal `in` with the filter kernel `coefficients_`
  // taking into account the previous state.
  for (size_t i = 0; i < length; ++i) {
    out[i] = 0.f;
    size_t j;
    for (j = 0; state_length_ > i && j < state_length_ - i; ++j) {
      out[i] += state_[i + j] * coefficients_[j];
    }
    for (; j < coefficients_length_; ++j) {
      out[i] += in[j + i - state_length_] * coefficients_[j];
    }
  }

  // Update current state.
  if (length >= state_length_) {
    memcpy(state_.get(), &in[length - state_length_],
           state_length_ * sizeof(*in));
  } else {
    memmove(state_.get(), &state_[length],
            (state_length_ - length) * sizeof(state_[0]));
    memcpy(&state_[state_length_ - length], in, length * sizeof(*in));
  }
}

}  // namespace webrtc
