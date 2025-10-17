/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#include "common_audio/fir_filter_neon.h"

#include <arm_neon.h>
#include <string.h>

#include "rtc_base/checks.h"
#include "rtc_base/memory/aligned_malloc.h"

namespace webrtc {

FIRFilterNEON::~FIRFilterNEON() {}

FIRFilterNEON::FIRFilterNEON(const float* coefficients,
                             size_t coefficients_length,
                             size_t max_input_length)
    :  // Closest higher multiple of four.
      coefficients_length_((coefficients_length + 3) & ~0x03),
      state_length_(coefficients_length_ - 1),
      coefficients_(static_cast<float*>(
          AlignedMalloc(sizeof(float) * coefficients_length_, 16))),
      state_(static_cast<float*>(
          AlignedMalloc(sizeof(float) * (max_input_length + state_length_),
                        16))) {
  // Add zeros at the end of the coefficients.
  size_t padding = coefficients_length_ - coefficients_length;
  memset(coefficients_.get(), 0.f, padding * sizeof(coefficients_[0]));
  // The coefficients are reversed to compensate for the order in which the
  // input samples are acquired (most recent last).
  for (size_t i = 0; i < coefficients_length; ++i) {
    coefficients_[i + padding] = coefficients[coefficients_length - i - 1];
  }
  memset(state_.get(), 0.f,
         (max_input_length + state_length_) * sizeof(state_[0]));
}

void FIRFilterNEON::Filter(const float* in, size_t length, float* out) {
  RTC_DCHECK_GT(length, 0);

  memcpy(&state_[state_length_], in, length * sizeof(*in));

  // Convolves the input signal `in` with the filter kernel `coefficients_`
  // taking into account the previous state.
  for (size_t i = 0; i < length; ++i) {
    float* in_ptr = &state_[i];
    float* coef_ptr = coefficients_.get();

    float32x4_t m_sum = vmovq_n_f32(0);
    float32x4_t m_in;

    for (size_t j = 0; j < coefficients_length_; j += 4) {
      m_in = vld1q_f32(in_ptr + j);
      m_sum = vmlaq_f32(m_sum, m_in, vld1q_f32(coef_ptr + j));
    }

    float32x2_t m_half = vadd_f32(vget_high_f32(m_sum), vget_low_f32(m_sum));
    out[i] = vget_lane_f32(vpadd_f32(m_half, m_half), 0);
  }

  // Update current state.
  memmove(state_.get(), &state_[length], state_length_ * sizeof(state_[0]));
}

}  // namespace webrtc
