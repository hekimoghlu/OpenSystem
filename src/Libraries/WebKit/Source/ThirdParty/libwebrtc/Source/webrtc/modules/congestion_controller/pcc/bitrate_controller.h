/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
#ifndef MODULES_CONGESTION_CONTROLLER_PCC_BITRATE_CONTROLLER_H_
#define MODULES_CONGESTION_CONTROLLER_PCC_BITRATE_CONTROLLER_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <vector>

#include "api/units/data_rate.h"
#include "modules/congestion_controller/pcc/monitor_interval.h"
#include "modules/congestion_controller/pcc/utility_function.h"

namespace webrtc {
namespace pcc {

class PccBitrateController {
 public:
  PccBitrateController(double initial_conversion_factor,
                       double initial_dynamic_boundary,
                       double dynamic_boundary_increment,
                       double rtt_gradient_coefficient,
                       double loss_coefficient,
                       double throughput_coefficient,
                       double throughput_power,
                       double rtt_gradient_threshold,
                       double delay_gradient_negative_bound);

  PccBitrateController(
      double initial_conversion_factor,
      double initial_dynamic_boundary,
      double dynamic_boundary_increment,
      std::unique_ptr<PccUtilityFunctionInterface> utility_function);

  std::optional<DataRate> ComputeRateUpdateForSlowStartMode(
      const PccMonitorInterval& monitor_interval);

  DataRate ComputeRateUpdateForOnlineLearningMode(
      const std::vector<PccMonitorInterval>& block,
      DataRate bandwidth_estimate);

  ~PccBitrateController();

 private:
  double ApplyDynamicBoundary(double rate_change, double bitrate);
  double ComputeStepSize(double utility_gradient);

  // Dynamic boundary variables:
  int64_t consecutive_boundary_adjustments_number_;
  const double initial_dynamic_boundary_;
  const double dynamic_boundary_increment_;

  const std::unique_ptr<PccUtilityFunctionInterface> utility_function_;
  // Step Size variables:
  int64_t step_size_adjustments_number_;
  const double initial_conversion_factor_;

  std::optional<double> previous_utility_;
};

}  // namespace pcc
}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_PCC_BITRATE_CONTROLLER_H_
