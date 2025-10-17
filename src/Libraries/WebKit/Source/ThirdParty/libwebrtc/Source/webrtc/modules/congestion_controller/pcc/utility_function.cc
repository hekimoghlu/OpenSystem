/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#include "modules/congestion_controller/pcc/utility_function.h"

#include <algorithm>
#include <cmath>

#include "api/units/data_rate.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace pcc {

VivaceUtilityFunction::VivaceUtilityFunction(
    double delay_gradient_coefficient,
    double loss_coefficient,
    double throughput_coefficient,
    double throughput_power,
    double delay_gradient_threshold,
    double delay_gradient_negative_bound)
    : delay_gradient_coefficient_(delay_gradient_coefficient),
      loss_coefficient_(loss_coefficient),
      throughput_power_(throughput_power),
      throughput_coefficient_(throughput_coefficient),
      delay_gradient_threshold_(delay_gradient_threshold),
      delay_gradient_negative_bound_(delay_gradient_negative_bound) {
  RTC_DCHECK_GE(delay_gradient_negative_bound_, 0);
}

double VivaceUtilityFunction::Compute(
    const PccMonitorInterval& monitor_interval) const {
  RTC_DCHECK(monitor_interval.IsFeedbackCollectionDone());
  double bitrate = monitor_interval.GetTargetSendingRate().bps();
  double loss_rate = monitor_interval.GetLossRate();
  double rtt_gradient =
      monitor_interval.ComputeDelayGradient(delay_gradient_threshold_);
  rtt_gradient = std::max(rtt_gradient, -delay_gradient_negative_bound_);
  return (throughput_coefficient_ * std::pow(bitrate, throughput_power_)) -
         (delay_gradient_coefficient_ * bitrate * rtt_gradient) -
         (loss_coefficient_ * bitrate * loss_rate);
}

VivaceUtilityFunction::~VivaceUtilityFunction() = default;

ModifiedVivaceUtilityFunction::ModifiedVivaceUtilityFunction(
    double delay_gradient_coefficient,
    double loss_coefficient,
    double throughput_coefficient,
    double throughput_power,
    double delay_gradient_threshold,
    double delay_gradient_negative_bound)
    : delay_gradient_coefficient_(delay_gradient_coefficient),
      loss_coefficient_(loss_coefficient),
      throughput_power_(throughput_power),
      throughput_coefficient_(throughput_coefficient),
      delay_gradient_threshold_(delay_gradient_threshold),
      delay_gradient_negative_bound_(delay_gradient_negative_bound) {
  RTC_DCHECK_GE(delay_gradient_negative_bound_, 0);
}

double ModifiedVivaceUtilityFunction::Compute(
    const PccMonitorInterval& monitor_interval) const {
  RTC_DCHECK(monitor_interval.IsFeedbackCollectionDone());
  double bitrate = monitor_interval.GetTargetSendingRate().bps();
  double loss_rate = monitor_interval.GetLossRate();
  double rtt_gradient =
      monitor_interval.ComputeDelayGradient(delay_gradient_threshold_);
  rtt_gradient = std::max(rtt_gradient, -delay_gradient_negative_bound_);
  return (throughput_coefficient_ * std::pow(bitrate, throughput_power_) *
          bitrate) -
         (delay_gradient_coefficient_ * bitrate * bitrate * rtt_gradient) -
         (loss_coefficient_ * bitrate * bitrate * loss_rate);
}

ModifiedVivaceUtilityFunction::~ModifiedVivaceUtilityFunction() = default;

}  // namespace pcc
}  // namespace webrtc
