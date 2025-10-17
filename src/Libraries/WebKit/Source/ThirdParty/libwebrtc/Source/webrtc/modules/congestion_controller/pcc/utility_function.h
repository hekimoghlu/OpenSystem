/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#ifndef MODULES_CONGESTION_CONTROLLER_PCC_UTILITY_FUNCTION_H_
#define MODULES_CONGESTION_CONTROLLER_PCC_UTILITY_FUNCTION_H_

#include "modules/congestion_controller/pcc/monitor_interval.h"

namespace webrtc {
namespace pcc {

// Utility function is used by PCC to transform the performance statistics
// (sending rate, loss rate, packets latency) gathered at one monitor interval
// into a numerical value.
// https://www.usenix.org/conference/nsdi18/presentation/dong
class PccUtilityFunctionInterface {
 public:
  virtual double Compute(const PccMonitorInterval& monitor_interval) const = 0;
  virtual ~PccUtilityFunctionInterface() = default;
};

// Vivace utility function were suggested in the paper "PCC Vivace:
// Online-Learning Congestion Control", Mo Dong et all.
class VivaceUtilityFunction : public PccUtilityFunctionInterface {
 public:
  VivaceUtilityFunction(double delay_gradient_coefficient,
                        double loss_coefficient,
                        double throughput_coefficient,
                        double throughput_power,
                        double delay_gradient_threshold,
                        double delay_gradient_negative_bound);
  double Compute(const PccMonitorInterval& monitor_interval) const override;
  ~VivaceUtilityFunction() override;

 private:
  const double delay_gradient_coefficient_;
  const double loss_coefficient_;
  const double throughput_power_;
  const double throughput_coefficient_;
  const double delay_gradient_threshold_;
  const double delay_gradient_negative_bound_;
};

// This utility function were obtained by tuning Vivace utility function.
// The main difference is that gradient of modified utilify funtion (as well as
// rate updates) scales proportionally to the sending rate which leads to
// better performance in case of single sender.
class ModifiedVivaceUtilityFunction : public PccUtilityFunctionInterface {
 public:
  ModifiedVivaceUtilityFunction(double delay_gradient_coefficient,
                                double loss_coefficient,
                                double throughput_coefficient,
                                double throughput_power,
                                double delay_gradient_threshold,
                                double delay_gradient_negative_bound);
  double Compute(const PccMonitorInterval& monitor_interval) const override;
  ~ModifiedVivaceUtilityFunction() override;

 private:
  const double delay_gradient_coefficient_;
  const double loss_coefficient_;
  const double throughput_power_;
  const double throughput_coefficient_;
  const double delay_gradient_threshold_;
  const double delay_gradient_negative_bound_;
};

}  // namespace pcc
}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_PCC_UTILITY_FUNCTION_H_
