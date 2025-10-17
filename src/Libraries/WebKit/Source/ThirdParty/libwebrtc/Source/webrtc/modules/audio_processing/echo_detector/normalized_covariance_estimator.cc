/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#include "modules/audio_processing/echo_detector/normalized_covariance_estimator.h"

#include <math.h>

#include "rtc_base/checks.h"

namespace webrtc {
namespace {

// Parameter controlling the adaptation speed.
constexpr float kAlpha = 0.001f;

}  // namespace

void NormalizedCovarianceEstimator::Update(float x,
                                           float x_mean,
                                           float x_sigma,
                                           float y,
                                           float y_mean,
                                           float y_sigma) {
  covariance_ =
      (1.f - kAlpha) * covariance_ + kAlpha * (x - x_mean) * (y - y_mean);
  normalized_cross_correlation_ = covariance_ / (x_sigma * y_sigma + .0001f);
  RTC_DCHECK(isfinite(covariance_));
  RTC_DCHECK(isfinite(normalized_cross_correlation_));
}

void NormalizedCovarianceEstimator::Clear() {
  covariance_ = 0.f;
  normalized_cross_correlation_ = 0.f;
}

}  // namespace webrtc
