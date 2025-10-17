/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#include "call/rtp_bitrate_configurator.h"

#include <algorithm>
#include <optional>

#include "api/transport/bitrate_settings.h"
#include "api/units/data_rate.h"
#include "rtc_base/checks.h"

namespace {

// Returns its smallest positive argument. If neither argument is positive,
// returns an arbitrary nonpositive value.
int MinPositive(int a, int b) {
  if (a <= 0) {
    return b;
  }
  if (b <= 0) {
    return a;
  }
  return std::min(a, b);
}

}  // namespace

namespace webrtc {
RtpBitrateConfigurator::RtpBitrateConfigurator(
    const BitrateConstraints& bitrate_config)
    : bitrate_config_(bitrate_config), base_bitrate_config_(bitrate_config) {
  RTC_DCHECK_GE(bitrate_config.min_bitrate_bps, 0);
  RTC_DCHECK_GE(bitrate_config.start_bitrate_bps,
                bitrate_config.min_bitrate_bps);
  if (bitrate_config.max_bitrate_bps != -1) {
    RTC_DCHECK_GE(bitrate_config.max_bitrate_bps,
                  bitrate_config.start_bitrate_bps);
  }
}

RtpBitrateConfigurator::~RtpBitrateConfigurator() = default;

BitrateConstraints RtpBitrateConfigurator::GetConfig() const {
  return bitrate_config_;
}

std::optional<BitrateConstraints>
RtpBitrateConfigurator::UpdateWithSdpParameters(
    const BitrateConstraints& bitrate_config) {
  RTC_DCHECK_GE(bitrate_config.min_bitrate_bps, 0);
  RTC_DCHECK_NE(bitrate_config.start_bitrate_bps, 0);
  if (bitrate_config.max_bitrate_bps != -1) {
    RTC_DCHECK_GT(bitrate_config.max_bitrate_bps, 0);
  }

  std::optional<int> new_start;
  // Only update the "start" bitrate if it's set, and different from the old
  // value. In practice, this value comes from the x-google-start-bitrate codec
  // parameter in SDP, and setting the same remote description twice shouldn't
  // restart bandwidth estimation.
  if (bitrate_config.start_bitrate_bps != -1 &&
      bitrate_config.start_bitrate_bps !=
          base_bitrate_config_.start_bitrate_bps) {
    new_start.emplace(bitrate_config.start_bitrate_bps);
  }
  base_bitrate_config_ = bitrate_config;
  return UpdateConstraints(new_start);
}

std::optional<BitrateConstraints>
RtpBitrateConfigurator::UpdateWithClientPreferences(
    const BitrateSettings& bitrate_mask) {
  bitrate_config_mask_ = bitrate_mask;
  return UpdateConstraints(bitrate_mask.start_bitrate_bps);
}

// Relay cap can change only max bitrate.
std::optional<BitrateConstraints> RtpBitrateConfigurator::UpdateWithRelayCap(
    DataRate cap) {
  if (cap.IsFinite()) {
    RTC_DCHECK(!cap.IsZero());
  }
  max_bitrate_over_relay_ = cap;
  return UpdateConstraints(std::nullopt);
}

std::optional<BitrateConstraints> RtpBitrateConfigurator::UpdateConstraints(
    const std::optional<int>& new_start) {
  BitrateConstraints updated;
  updated.min_bitrate_bps =
      std::max(bitrate_config_mask_.min_bitrate_bps.value_or(0),
               base_bitrate_config_.min_bitrate_bps);

  updated.max_bitrate_bps =
      MinPositive(bitrate_config_mask_.max_bitrate_bps.value_or(-1),
                  base_bitrate_config_.max_bitrate_bps);
  updated.max_bitrate_bps =
      MinPositive(updated.max_bitrate_bps, max_bitrate_over_relay_.bps_or(-1));

  // If the combined min ends up greater than the combined max, the max takes
  // priority.
  if (updated.max_bitrate_bps != -1 &&
      updated.min_bitrate_bps > updated.max_bitrate_bps) {
    updated.min_bitrate_bps = updated.max_bitrate_bps;
  }

  // If there is nothing to update (min/max unchanged, no new bandwidth
  // estimation start value), return early.
  if (updated.min_bitrate_bps == bitrate_config_.min_bitrate_bps &&
      updated.max_bitrate_bps == bitrate_config_.max_bitrate_bps &&
      !new_start) {
    return std::nullopt;
  }

  if (new_start) {
    // Clamp start by min and max.
    updated.start_bitrate_bps = MinPositive(
        std::max(*new_start, updated.min_bitrate_bps), updated.max_bitrate_bps);
  } else {
    updated.start_bitrate_bps = -1;
  }
  BitrateConstraints config_to_return = updated;
  if (!new_start) {
    updated.start_bitrate_bps = bitrate_config_.start_bitrate_bps;
  }
  bitrate_config_ = updated;
  return config_to_return;
}

}  // namespace webrtc
