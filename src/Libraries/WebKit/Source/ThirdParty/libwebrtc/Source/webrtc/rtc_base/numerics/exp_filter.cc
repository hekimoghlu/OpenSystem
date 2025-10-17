/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#include "rtc_base/numerics/exp_filter.h"

#include <cmath>

namespace rtc {

const float ExpFilter::kValueUndefined = -1.0f;

void ExpFilter::Reset(float alpha) {
  alpha_ = alpha;
  filtered_ = kValueUndefined;
}

float ExpFilter::Apply(float exp, float sample) {
  if (filtered_ == kValueUndefined) {
    // Initialize filtered value.
    filtered_ = sample;
  } else if (exp == 1.0) {
    filtered_ = alpha_ * filtered_ + (1 - alpha_) * sample;
  } else {
    float alpha = std::pow(alpha_, exp);
    filtered_ = alpha * filtered_ + (1 - alpha) * sample;
  }
  if (max_ != kValueUndefined && filtered_ > max_) {
    filtered_ = max_;
  }
  return filtered_;
}

void ExpFilter::UpdateBase(float alpha) {
  alpha_ = alpha;
}
}  // namespace rtc
