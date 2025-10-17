/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#ifndef MODULES_CONGESTION_CONTROLLER_GOOG_CC_ROBUST_THROUGHPUT_ESTIMATOR_H_
#define MODULES_CONGESTION_CONTROLLER_GOOG_CC_ROBUST_THROUGHPUT_ESTIMATOR_H_

#include <deque>
#include <optional>
#include <vector>

#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/timestamp.h"
#include "modules/congestion_controller/goog_cc/acknowledged_bitrate_estimator_interface.h"

namespace webrtc {

class RobustThroughputEstimator : public AcknowledgedBitrateEstimatorInterface {
 public:
  explicit RobustThroughputEstimator(
      const RobustThroughputEstimatorSettings& settings);
  ~RobustThroughputEstimator() override;

  void IncomingPacketFeedbackVector(
      const std::vector<PacketResult>& packet_feedback_vector) override;

  std::optional<DataRate> bitrate() const override;

  std::optional<DataRate> PeekRate() const override { return bitrate(); }
  void SetAlr(bool /*in_alr*/) override {}
  void SetAlrEndedTime(Timestamp /*alr_ended_time*/) override {}

 private:
  bool FirstPacketOutsideWindow();

  const RobustThroughputEstimatorSettings settings_;
  std::deque<PacketResult> window_;
  Timestamp latest_discarded_send_time_ = Timestamp::MinusInfinity();
};

}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_GOOG_CC_ROBUST_THROUGHPUT_ESTIMATOR_H_
