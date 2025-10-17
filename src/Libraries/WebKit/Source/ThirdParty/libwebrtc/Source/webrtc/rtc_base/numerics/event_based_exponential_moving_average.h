/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
#ifndef RTC_BASE_NUMERICS_EVENT_BASED_EXPONENTIAL_MOVING_AVERAGE_H_
#define RTC_BASE_NUMERICS_EVENT_BASED_EXPONENTIAL_MOVING_AVERAGE_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace rtc {

/**
 * This class implements exponential moving average for time series
 * estimating both value, variance and variance of estimator based on
 * https://en.wikipedia.org/w/index.php?title=Moving_average&section=9#Application_to_measuring_computer_performance
 * with the additions from nisse@ added to
 * https://en.wikipedia.org/wiki/Talk:Moving_average.
 *
 * A sample gets exponentially less weight so that it's 50%
 * after `half_time` time units.
 */
class EventBasedExponentialMovingAverage {
 public:
  // `half_time` specifies how much weight will be given to old samples,
  // see example above.
  explicit EventBasedExponentialMovingAverage(int half_time);

  void AddSample(int64_t now, int value);

  double GetAverage() const { return value_; }
  double GetVariance() const { return sample_variance_; }

  // Compute 95% confidence interval assuming that
  // - variance of samples are normal distributed.
  // - variance of estimator is normal distributed.
  //
  // The returned values specifies the distance from the average,
  // i.e if X = GetAverage(), m = GetConfidenceInterval()
  // then a there is 95% likelihood that the observed variables is inside
  // [ X +/- m ].
  double GetConfidenceInterval() const;

  // Reset
  void Reset();

  // Update the half_time.
  // NOTE: resets estimate too.
  void SetHalfTime(int half_time);

 private:
  double tau_;
  double value_ = std::nan("uninit");
  double sample_variance_ = std::numeric_limits<double>::infinity();
  // This is the ratio between variance of the estimate and variance of samples.
  double estimator_variance_ = 1;
  std::optional<int64_t> last_observation_timestamp_;
};

}  // namespace rtc

#endif  // RTC_BASE_NUMERICS_EVENT_BASED_EXPONENTIAL_MOVING_AVERAGE_H_
