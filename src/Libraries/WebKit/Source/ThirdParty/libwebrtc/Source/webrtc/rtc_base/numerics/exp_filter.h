/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#ifndef RTC_BASE_NUMERICS_EXP_FILTER_H_
#define RTC_BASE_NUMERICS_EXP_FILTER_H_

namespace rtc {

// This class can be used, for example, for smoothing the result of bandwidth
// estimation and packet loss estimation.

class ExpFilter {
 public:
  static const float kValueUndefined;

  explicit ExpFilter(float alpha, float max = kValueUndefined) : max_(max) {
    Reset(alpha);
  }

  // Resets the filter to its initial state, and resets filter factor base to
  // the given value `alpha`.
  void Reset(float alpha);

  // Applies the filter with a given exponent on the provided sample:
  // y(k) = min(alpha_^ exp * y(k-1) + (1 - alpha_^ exp) * sample, max_).
  float Apply(float exp, float sample);

  // Returns current filtered value.
  float filtered() const { return filtered_; }

  // Changes the filter factor base to the given value `alpha`.
  void UpdateBase(float alpha);

 private:
  float alpha_;     // Filter factor base.
  float filtered_;  // Current filter output.
  const float max_;
};
}  // namespace rtc

#endif  // RTC_BASE_NUMERICS_EXP_FILTER_H_
