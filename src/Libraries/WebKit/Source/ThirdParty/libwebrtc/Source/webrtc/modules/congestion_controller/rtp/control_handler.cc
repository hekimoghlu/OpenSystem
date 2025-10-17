/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#include "modules/congestion_controller/rtp/control_handler.h"

#include <algorithm>
#include <vector>

#include "api/units/data_rate.h"
#include "modules/pacing/pacing_controller.h"
#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "rtc_base/numerics/safe_minmax.h"

namespace webrtc {

void CongestionControlHandler::SetTargetRate(
    TargetTransferRate new_target_rate) {
  RTC_DCHECK_RUN_ON(&sequenced_checker_);
  RTC_CHECK(new_target_rate.at_time.IsFinite());
  last_incoming_ = new_target_rate;
}

void CongestionControlHandler::SetNetworkAvailability(bool network_available) {
  RTC_DCHECK_RUN_ON(&sequenced_checker_);
  network_available_ = network_available;
}

void CongestionControlHandler::SetPacerQueue(TimeDelta expected_queue_time) {
  RTC_DCHECK_RUN_ON(&sequenced_checker_);
  pacer_expected_queue_ms_ = expected_queue_time.ms();
}

std::optional<TargetTransferRate> CongestionControlHandler::GetUpdate() {
  RTC_DCHECK_RUN_ON(&sequenced_checker_);
  if (!last_incoming_.has_value())
    return std::nullopt;
  TargetTransferRate new_outgoing = *last_incoming_;
  DataRate log_target_rate = new_outgoing.target_rate;
  bool pause_encoding = false;
  if (!network_available_) {
    pause_encoding = true;
  } else if (pacer_expected_queue_ms_ >
             PacingController::kMaxExpectedQueueLength.ms()) {
    pause_encoding = true;
  }
  if (pause_encoding)
    new_outgoing.target_rate = DataRate::Zero();
  if (!last_reported_ ||
      last_reported_->target_rate != new_outgoing.target_rate ||
      (!new_outgoing.target_rate.IsZero() &&
       (last_reported_->network_estimate.loss_rate_ratio !=
            new_outgoing.network_estimate.loss_rate_ratio ||
        last_reported_->network_estimate.round_trip_time !=
            new_outgoing.network_estimate.round_trip_time))) {
    if (encoder_paused_in_last_report_ != pause_encoding)
      RTC_LOG(LS_INFO) << "Bitrate estimate state changed, BWE: "
                       << ToString(log_target_rate) << ".";
    encoder_paused_in_last_report_ = pause_encoding;
    last_reported_ = new_outgoing;
    return new_outgoing;
  }
  return std::nullopt;
}

}  // namespace webrtc
