/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_VAD_POLE_ZERO_FILTER_H_
#define MODULES_AUDIO_PROCESSING_VAD_POLE_ZERO_FILTER_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {

class PoleZeroFilter {
 public:
  ~PoleZeroFilter() {}

  static PoleZeroFilter* Create(const float* numerator_coefficients,
                                size_t order_numerator,
                                const float* denominator_coefficients,
                                size_t order_denominator);

  int Filter(const int16_t* in, size_t num_input_samples, float* output);

 private:
  PoleZeroFilter(const float* numerator_coefficients,
                 size_t order_numerator,
                 const float* denominator_coefficients,
                 size_t order_denominator);

  static const int kMaxFilterOrder = 24;

  int16_t past_input_[kMaxFilterOrder * 2];
  float past_output_[kMaxFilterOrder * 2];

  float numerator_coefficients_[kMaxFilterOrder + 1];
  float denominator_coefficients_[kMaxFilterOrder + 1];

  size_t order_numerator_;
  size_t order_denominator_;
  size_t highest_order_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_VAD_POLE_ZERO_FILTER_H_
