/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_COMPUTE_INTERPOLATED_GAIN_CURVE_H_
#define MODULES_AUDIO_PROCESSING_AGC2_COMPUTE_INTERPOLATED_GAIN_CURVE_H_

#include <array>

#include "modules/audio_processing/agc2/agc2_common.h"

namespace webrtc {

namespace test {

// Parameters for interpolated gain curve using under-approximation to
// avoid saturation.
//
// The saturation gain is defined in order to let hard-clipping occur for
// those samples having a level that falls in the saturation region. It is an
// upper bound of the actual gain to apply - i.e., that returned by the
// limiter.

// Knee and beyond-knee regions approximation parameters.
// The gain curve is approximated as a piece-wise linear function.
// `approx_params_x_` are the boundaries between adjacent linear pieces,
// `approx_params_m_` and `approx_params_q_` are the slope and the y-intercept
// values of each piece.
struct InterpolatedParameters {
  std::array<float, kInterpolatedGainCurveTotalPoints>
      computed_approximation_params_x;
  std::array<float, kInterpolatedGainCurveTotalPoints>
      computed_approximation_params_m;
  std::array<float, kInterpolatedGainCurveTotalPoints>
      computed_approximation_params_q;
};

InterpolatedParameters ComputeInterpolatedGainCurveApproximationParams();
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_COMPUTE_INTERPOLATED_GAIN_CURVE_H_
