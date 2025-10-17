/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#include "modules/audio_processing/aec3/subtractor_output.h"

#include <numeric>

namespace webrtc {

SubtractorOutput::SubtractorOutput() = default;
SubtractorOutput::~SubtractorOutput() = default;

void SubtractorOutput::Reset() {
  s_refined.fill(0.f);
  s_coarse.fill(0.f);
  e_refined.fill(0.f);
  e_coarse.fill(0.f);
  E_refined.re.fill(0.f);
  E_refined.im.fill(0.f);
  E2_refined.fill(0.f);
  E2_coarse.fill(0.f);
  e2_refined = 0.f;
  e2_coarse = 0.f;
  s2_refined = 0.f;
  s2_coarse = 0.f;
  y2 = 0.f;
}

void SubtractorOutput::ComputeMetrics(rtc::ArrayView<const float> y) {
  const auto sum_of_squares = [](float a, float b) { return a + b * b; };
  y2 = std::accumulate(y.begin(), y.end(), 0.f, sum_of_squares);
  e2_refined =
      std::accumulate(e_refined.begin(), e_refined.end(), 0.f, sum_of_squares);
  e2_coarse =
      std::accumulate(e_coarse.begin(), e_coarse.end(), 0.f, sum_of_squares);
  s2_refined =
      std::accumulate(s_refined.begin(), s_refined.end(), 0.f, sum_of_squares);
  s2_coarse =
      std::accumulate(s_coarse.begin(), s_coarse.end(), 0.f, sum_of_squares);

  s_refined_max_abs = *std::max_element(s_refined.begin(), s_refined.end());
  s_refined_max_abs =
      std::max(s_refined_max_abs,
               -(*std::min_element(s_refined.begin(), s_refined.end())));

  s_coarse_max_abs = *std::max_element(s_coarse.begin(), s_coarse.end());
  s_coarse_max_abs = std::max(
      s_coarse_max_abs, -(*std::min_element(s_coarse.begin(), s_coarse.end())));
}

}  // namespace webrtc
