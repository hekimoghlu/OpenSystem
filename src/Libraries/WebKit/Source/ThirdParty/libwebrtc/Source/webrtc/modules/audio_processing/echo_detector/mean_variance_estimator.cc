/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#include "modules/audio_processing/echo_detector/mean_variance_estimator.h"

#include <math.h>

#include "rtc_base/checks.h"

namespace webrtc {
namespace {

// Parameter controlling the adaptation speed.
constexpr float kAlpha = 0.001f;

}  // namespace

void MeanVarianceEstimator::Update(float value) {
  mean_ = (1.f - kAlpha) * mean_ + kAlpha * value;
  variance_ =
      (1.f - kAlpha) * variance_ + kAlpha * (value - mean_) * (value - mean_);
  RTC_DCHECK(isfinite(mean_));
  RTC_DCHECK(isfinite(variance_));
}

float MeanVarianceEstimator::std_deviation() const {
  RTC_DCHECK_GE(variance_, 0.f);
  return sqrtf(variance_);
}

float MeanVarianceEstimator::mean() const {
  return mean_;
}

void MeanVarianceEstimator::Clear() {
  mean_ = 0.f;
  variance_ = 0.f;
}

}  // namespace webrtc
