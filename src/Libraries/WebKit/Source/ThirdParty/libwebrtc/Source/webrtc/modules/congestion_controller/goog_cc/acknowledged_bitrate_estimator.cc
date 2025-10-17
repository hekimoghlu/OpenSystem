/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#include "modules/congestion_controller/goog_cc/acknowledged_bitrate_estimator.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "api/field_trials_view.h"
#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/timestamp.h"
#include "modules/congestion_controller/goog_cc/bitrate_estimator.h"
#include "rtc_base/checks.h"

namespace webrtc {

AcknowledgedBitrateEstimator::AcknowledgedBitrateEstimator(
    const FieldTrialsView* key_value_config)
    : AcknowledgedBitrateEstimator(
          key_value_config,
          std::make_unique<BitrateEstimator>(key_value_config)) {}

AcknowledgedBitrateEstimator::~AcknowledgedBitrateEstimator() {}

AcknowledgedBitrateEstimator::AcknowledgedBitrateEstimator(
    const FieldTrialsView* /* key_value_config */,
    std::unique_ptr<BitrateEstimator> bitrate_estimator)
    : in_alr_(false), bitrate_estimator_(std::move(bitrate_estimator)) {}

void AcknowledgedBitrateEstimator::IncomingPacketFeedbackVector(
    const std::vector<PacketResult>& packet_feedback_vector) {
  RTC_DCHECK(std::is_sorted(packet_feedback_vector.begin(),
                            packet_feedback_vector.end(),
                            PacketResult::ReceiveTimeOrder()));
  for (const auto& packet : packet_feedback_vector) {
    if (alr_ended_time_ && packet.sent_packet.send_time > *alr_ended_time_) {
      bitrate_estimator_->ExpectFastRateChange();
      alr_ended_time_.reset();
    }
    DataSize acknowledged_estimate = packet.sent_packet.size;
    acknowledged_estimate += packet.sent_packet.prior_unacked_data;
    bitrate_estimator_->Update(packet.receive_time, acknowledged_estimate,
                               in_alr_);
  }
}

std::optional<DataRate> AcknowledgedBitrateEstimator::bitrate() const {
  return bitrate_estimator_->bitrate();
}

std::optional<DataRate> AcknowledgedBitrateEstimator::PeekRate() const {
  return bitrate_estimator_->PeekRate();
}

void AcknowledgedBitrateEstimator::SetAlrEndedTime(Timestamp alr_ended_time) {
  alr_ended_time_.emplace(alr_ended_time);
}

void AcknowledgedBitrateEstimator::SetAlr(bool in_alr) {
  in_alr_ = in_alr;
}

}  // namespace webrtc
