/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#ifndef MODULES_REMOTE_BITRATE_ESTIMATOR_AIMD_RATE_CONTROL_H_
#define MODULES_REMOTE_BITRATE_ESTIMATOR_AIMD_RATE_CONTROL_H_

#include <stdint.h>

#include <optional>

#include "api/field_trials_view.h"
#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/timestamp.h"
#include "modules/congestion_controller/goog_cc/link_capacity_estimator.h"
#include "modules/remote_bitrate_estimator/include/bwe_defines.h"
#include "rtc_base/experiments/field_trial_parser.h"

namespace webrtc {
// A rate control implementation based on additive increases of
// bitrate when no over-use is detected and multiplicative decreases when
// over-uses are detected. When we think the available bandwidth has changes or
// is unknown, we will switch to a "slow-start mode" where we increase
// multiplicatively.
class AimdRateControl {
 public:
  explicit AimdRateControl(const FieldTrialsView& key_value_config);
  AimdRateControl(const FieldTrialsView& key_value_config, bool send_side);
  ~AimdRateControl();

  // Returns true if the target bitrate has been initialized. This happens
  // either if it has been explicitly set via SetStartBitrate/SetEstimate, or if
  // we have measured a throughput.
  bool ValidEstimate() const;
  void SetStartBitrate(DataRate start_bitrate);
  void SetMinBitrate(DataRate min_bitrate);
  TimeDelta GetFeedbackInterval() const;

  // Returns true if the bitrate estimate hasn't been changed for more than
  // an RTT, or if the estimated_throughput is less than half of the current
  // estimate. Should be used to decide if we should reduce the rate further
  // when over-using.
  bool TimeToReduceFurther(Timestamp at_time,
                           DataRate estimated_throughput) const;
  // As above. To be used if overusing before we have measured a throughput.
  bool InitialTimeToReduceFurther(Timestamp at_time) const;

  DataRate LatestEstimate() const;
  void SetRtt(TimeDelta rtt);
  DataRate Update(const RateControlInput& input, Timestamp at_time);
  void SetInApplicationLimitedRegion(bool in_alr);
  void SetEstimate(DataRate bitrate, Timestamp at_time);
  void SetNetworkStateEstimate(
      const std::optional<NetworkStateEstimate>& estimate);

  // Returns the increase rate when used bandwidth is near the link capacity.
  double GetNearMaxIncreaseRateBpsPerSecond() const;
  // Returns the expected time between overuse signals (assuming steady state).
  TimeDelta GetExpectedBandwidthPeriod() const;

 private:
  enum class RateControlState { kRcHold, kRcIncrease, kRcDecrease };

  friend class GoogCcStatePrinter;
  // Update the target bitrate based on, among other things, the current rate
  // control state, the current target bitrate and the estimated throughput.
  // When in the "increase" state the bitrate will be increased either
  // additively or multiplicatively depending on the rate control region. When
  // in the "decrease" state the bitrate will be decreased to slightly below the
  // current throughput. When in the "hold" state the bitrate will be kept
  // constant to allow built up queues to drain.
  void ChangeBitrate(const RateControlInput& input, Timestamp at_time);

  DataRate ClampBitrate(DataRate new_bitrate) const;
  DataRate MultiplicativeRateIncrease(Timestamp at_time,
                                      Timestamp last_ms,
                                      DataRate current_bitrate) const;
  DataRate AdditiveRateIncrease(Timestamp at_time, Timestamp last_time) const;
  void UpdateChangePeriod(Timestamp at_time);
  void ChangeState(const RateControlInput& input, Timestamp at_time);

  DataRate min_configured_bitrate_;
  DataRate max_configured_bitrate_;
  DataRate current_bitrate_;
  DataRate latest_estimated_throughput_;
  LinkCapacityEstimator link_capacity_;
  std::optional<NetworkStateEstimate> network_estimate_;
  RateControlState rate_control_state_;
  Timestamp time_last_bitrate_change_;
  Timestamp time_last_bitrate_decrease_;
  Timestamp time_first_throughput_estimate_;
  bool bitrate_is_initialized_;
  double beta_;
  bool in_alr_;
  TimeDelta rtt_;
  const bool send_side_;
  // Allow the delay based estimate to only increase as long as application
  // limited region (alr) is not detected.
  const bool no_bitrate_increase_in_alr_;
  // If "Disabled",  estimated link capacity is not used as upper bound.
  FieldTrialFlag disable_estimate_bounded_increase_{"Disabled"};
  FieldTrialParameter<bool> use_current_estimate_as_min_upper_bound_{"c_upper",
                                                                     true};
  std::optional<DataRate> last_decrease_;
};
}  // namespace webrtc

#endif  // MODULES_REMOTE_BITRATE_ESTIMATOR_AIMD_RATE_CONTROL_H_
