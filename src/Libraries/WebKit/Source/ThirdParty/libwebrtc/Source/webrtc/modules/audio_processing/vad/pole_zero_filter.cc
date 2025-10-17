/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#include "modules/audio_processing/vad/pole_zero_filter.h"

#include <string.h>

#include <algorithm>

namespace webrtc {

PoleZeroFilter* PoleZeroFilter::Create(const float* numerator_coefficients,
                                       size_t order_numerator,
                                       const float* denominator_coefficients,
                                       size_t order_denominator) {
  if (order_numerator > kMaxFilterOrder ||
      order_denominator > kMaxFilterOrder || denominator_coefficients[0] == 0 ||
      numerator_coefficients == NULL || denominator_coefficients == NULL)
    return NULL;
  return new PoleZeroFilter(numerator_coefficients, order_numerator,
                            denominator_coefficients, order_denominator);
}

PoleZeroFilter::PoleZeroFilter(const float* numerator_coefficients,
                               size_t order_numerator,
                               const float* denominator_coefficients,
                               size_t order_denominator)
    : past_input_(),
      past_output_(),
      numerator_coefficients_(),
      denominator_coefficients_(),
      order_numerator_(order_numerator),
      order_denominator_(order_denominator),
      highest_order_(std::max(order_denominator, order_numerator)) {
  memcpy(numerator_coefficients_, numerator_coefficients,
         sizeof(numerator_coefficients_[0]) * (order_numerator_ + 1));
  memcpy(denominator_coefficients_, denominator_coefficients,
         sizeof(denominator_coefficients_[0]) * (order_denominator_ + 1));

  if (denominator_coefficients_[0] != 1) {
    for (size_t n = 0; n <= order_numerator_; n++)
      numerator_coefficients_[n] /= denominator_coefficients_[0];
    for (size_t n = 0; n <= order_denominator_; n++)
      denominator_coefficients_[n] /= denominator_coefficients_[0];
  }
}

template <typename T>
static float FilterArPast(const T* past,
                          size_t order,
                          const float* coefficients) {
  float sum = 0.0f;
  size_t past_index = order - 1;
  for (size_t k = 1; k <= order; k++, past_index--)
    sum += coefficients[k] * past[past_index];
  return sum;
}

int PoleZeroFilter::Filter(const int16_t* in,
                           size_t num_input_samples,
                           float* output) {
  if (in == NULL || output == NULL)
    return -1;
  // This is the typical case, just a memcpy.
  const size_t k = std::min(num_input_samples, highest_order_);
  size_t n;
  for (n = 0; n < k; n++) {
    output[n] = in[n] * numerator_coefficients_[0];
    output[n] += FilterArPast(&past_input_[n], order_numerator_,
                              numerator_coefficients_);
    output[n] -= FilterArPast(&past_output_[n], order_denominator_,
                              denominator_coefficients_);

    past_input_[n + order_numerator_] = in[n];
    past_output_[n + order_denominator_] = output[n];
  }
  if (highest_order_ < num_input_samples) {
    for (size_t m = 0; n < num_input_samples; n++, m++) {
      output[n] = in[n] * numerator_coefficients_[0];
      output[n] +=
          FilterArPast(&in[m], order_numerator_, numerator_coefficients_);
      output[n] -= FilterArPast(&output[m], order_denominator_,
                                denominator_coefficients_);
    }
    // Record into the past signal.
    memcpy(past_input_, &in[num_input_samples - order_numerator_],
           sizeof(in[0]) * order_numerator_);
    memcpy(past_output_, &output[num_input_samples - order_denominator_],
           sizeof(output[0]) * order_denominator_);
  } else {
    // Odd case that the length of the input is shorter that filter order.
    memmove(past_input_, &past_input_[num_input_samples],
            order_numerator_ * sizeof(past_input_[0]));
    memmove(past_output_, &past_output_[num_input_samples],
            order_denominator_ * sizeof(past_output_[0]));
  }
  return 0;
}

}  // namespace webrtc
