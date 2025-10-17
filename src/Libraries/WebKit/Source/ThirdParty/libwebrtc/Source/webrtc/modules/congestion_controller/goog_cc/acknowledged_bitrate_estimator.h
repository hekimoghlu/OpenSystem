/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
#ifndef MODULES_CONGESTION_CONTROLLER_GOOG_CC_ACKNOWLEDGED_BITRATE_ESTIMATOR_H_
#define MODULES_CONGESTION_CONTROLLER_GOOG_CC_ACKNOWLEDGED_BITRATE_ESTIMATOR_H_

#include <memory>
#include <optional>
#include <vector>

#include "api/field_trials_view.h"
#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/timestamp.h"
#include "modules/congestion_controller/goog_cc/acknowledged_bitrate_estimator_interface.h"
#include "modules/congestion_controller/goog_cc/bitrate_estimator.h"

namespace webrtc {

class AcknowledgedBitrateEstimator
    : public AcknowledgedBitrateEstimatorInterface {
 public:
  AcknowledgedBitrateEstimator(
      const FieldTrialsView* key_value_config,
      std::unique_ptr<BitrateEstimator> bitrate_estimator);

  explicit AcknowledgedBitrateEstimator(
      const FieldTrialsView* key_value_config);
  ~AcknowledgedBitrateEstimator() override;

  void IncomingPacketFeedbackVector(
      const std::vector<PacketResult>& packet_feedback_vector) override;
  std::optional<DataRate> bitrate() const override;
  std::optional<DataRate> PeekRate() const override;
  void SetAlr(bool in_alr) override;
  void SetAlrEndedTime(Timestamp alr_ended_time) override;

 private:
  std::optional<Timestamp> alr_ended_time_;
  bool in_alr_;
  std::unique_ptr<BitrateEstimator> bitrate_estimator_;
};

}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_GOOG_CC_ACKNOWLEDGED_BITRATE_ESTIMATOR_H_
