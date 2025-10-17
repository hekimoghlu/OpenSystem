/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
#include "modules/congestion_controller/goog_cc/congestion_window_pushback_controller.h"

#include <algorithm>
#include <cstdint>

#include "api/field_trials_view.h"
#include "api/units/data_size.h"
#include "rtc_base/experiments/rate_control_settings.h"

namespace webrtc {

CongestionWindowPushbackController::CongestionWindowPushbackController(
    const FieldTrialsView& key_value_config)
    : add_pacing_(key_value_config.IsEnabled(
          "WebRTC-AddPacingToCongestionWindowPushback")),
      min_pushback_target_bitrate_bps_(
          RateControlSettings(key_value_config)
              .CongestionWindowMinPushbackTargetBitrateBps()),
      current_data_window_(RateControlSettings(key_value_config)
                               .CongestionWindowInitialDataWindow()) {}

void CongestionWindowPushbackController::UpdateOutstandingData(
    int64_t outstanding_bytes) {
  outstanding_bytes_ = outstanding_bytes;
}
void CongestionWindowPushbackController::UpdatePacingQueue(
    int64_t pacing_bytes) {
  pacing_bytes_ = pacing_bytes;
}

void CongestionWindowPushbackController::SetDataWindow(DataSize data_window) {
  current_data_window_ = data_window;
}

uint32_t CongestionWindowPushbackController::UpdateTargetBitrate(
    uint32_t bitrate_bps) {
  if (!current_data_window_ || current_data_window_->IsZero())
    return bitrate_bps;
  int64_t total_bytes = outstanding_bytes_;
  if (add_pacing_)
    total_bytes += pacing_bytes_;
  double fill_ratio =
      total_bytes / static_cast<double>(current_data_window_->bytes());
  if (fill_ratio > 1.5) {
    encoding_rate_ratio_ *= 0.9;
  } else if (fill_ratio > 1) {
    encoding_rate_ratio_ *= 0.95;
  } else if (fill_ratio < 0.1) {
    encoding_rate_ratio_ = 1.0;
  } else {
    encoding_rate_ratio_ *= 1.05;
    encoding_rate_ratio_ = std::min(encoding_rate_ratio_, 1.0);
  }
  uint32_t adjusted_target_bitrate_bps =
      static_cast<uint32_t>(bitrate_bps * encoding_rate_ratio_);

  // Do not adjust below the minimum pushback bitrate but do obey if the
  // original estimate is below it.
  bitrate_bps = adjusted_target_bitrate_bps < min_pushback_target_bitrate_bps_
                    ? std::min(bitrate_bps, min_pushback_target_bitrate_bps_)
                    : adjusted_target_bitrate_bps;
  return bitrate_bps;
}

}  // namespace webrtc
