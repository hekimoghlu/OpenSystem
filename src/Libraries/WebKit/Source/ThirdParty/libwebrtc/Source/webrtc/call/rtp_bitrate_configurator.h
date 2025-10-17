/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
#ifndef CALL_RTP_BITRATE_CONFIGURATOR_H_
#define CALL_RTP_BITRATE_CONFIGURATOR_H_

#include <optional>

#include "api/transport/bitrate_settings.h"
#include "api/units/data_rate.h"

namespace webrtc {

// RtpBitrateConfigurator calculates the bitrate configuration based on received
// remote configuration combined with local overrides.
class RtpBitrateConfigurator {
 public:
  explicit RtpBitrateConfigurator(const BitrateConstraints& bitrate_config);
  ~RtpBitrateConfigurator();

  RtpBitrateConfigurator(const RtpBitrateConfigurator&) = delete;
  RtpBitrateConfigurator& operator=(const RtpBitrateConfigurator&) = delete;

  BitrateConstraints GetConfig() const;

  // The greater min and smaller max set by this and SetClientBitratePreferences
  // will be used. The latest non-negative start value from either call will be
  // used. Specifying a start bitrate (>0) will reset the current bitrate
  // estimate. This is due to how the 'x-google-start-bitrate' flag is currently
  // implemented. Passing -1 leaves the start bitrate unchanged. Behavior is not
  // guaranteed for other negative values or 0.
  // The optional return value is set with new configuration if it was updated.
  std::optional<BitrateConstraints> UpdateWithSdpParameters(
      const BitrateConstraints& bitrate_config_);

  // The greater min and smaller max set by this and SetSdpBitrateParameters
  // will be used. The latest non-negative start value form either call will be
  // used. Specifying a start bitrate will reset the current bitrate estimate.
  // Assumes 0 <= min <= start <= max holds for set parameters.
  // Update the bitrate configuration
  // The optional return value is set with new configuration if it was updated.
  std::optional<BitrateConstraints> UpdateWithClientPreferences(
      const BitrateSettings& bitrate_mask);

  // Apply a cap for relayed calls.
  std::optional<BitrateConstraints> UpdateWithRelayCap(DataRate cap);

 private:
  // Applies update to the BitrateConstraints cached in `config_`, resetting
  // with `new_start` if set.
  std::optional<BitrateConstraints> UpdateConstraints(
      const std::optional<int>& new_start);

  // Bitrate config used until valid bitrate estimates are calculated. Also
  // used to cap total bitrate used. This comes from the remote connection.
  BitrateConstraints bitrate_config_;

  // The config mask set by SetClientBitratePreferences.
  // 0 <= min <= start <= max
  BitrateSettings bitrate_config_mask_;

  // The config set by SetSdpBitrateParameters.
  // min >= 0, start != 0, max == -1 || max > 0
  BitrateConstraints base_bitrate_config_;

  // Bandwidth cap applied for relayed calls.
  DataRate max_bitrate_over_relay_ = DataRate::PlusInfinity();
};
}  // namespace webrtc

#endif  // CALL_RTP_BITRATE_CONFIGURATOR_H_
