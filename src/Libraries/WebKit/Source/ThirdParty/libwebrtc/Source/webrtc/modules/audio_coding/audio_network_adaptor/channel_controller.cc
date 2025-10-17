/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#include "modules/audio_coding/audio_network_adaptor/channel_controller.h"

#include <algorithm>

#include "rtc_base/checks.h"

namespace webrtc {

ChannelController::Config::Config(size_t num_encoder_channels,
                                  size_t intial_channels_to_encode,
                                  int channel_1_to_2_bandwidth_bps,
                                  int channel_2_to_1_bandwidth_bps)
    : num_encoder_channels(num_encoder_channels),
      intial_channels_to_encode(intial_channels_to_encode),
      channel_1_to_2_bandwidth_bps(channel_1_to_2_bandwidth_bps),
      channel_2_to_1_bandwidth_bps(channel_2_to_1_bandwidth_bps) {}

ChannelController::ChannelController(const Config& config)
    : config_(config), channels_to_encode_(config_.intial_channels_to_encode) {
  RTC_DCHECK_GT(config_.intial_channels_to_encode, 0lu);
  // Currently, we require `intial_channels_to_encode` to be <= 2.
  RTC_DCHECK_LE(config_.intial_channels_to_encode, 2lu);
  RTC_DCHECK_GE(config_.num_encoder_channels,
                config_.intial_channels_to_encode);
}

ChannelController::~ChannelController() = default;

void ChannelController::UpdateNetworkMetrics(
    const NetworkMetrics& network_metrics) {
  if (network_metrics.uplink_bandwidth_bps)
    uplink_bandwidth_bps_ = network_metrics.uplink_bandwidth_bps;
}

void ChannelController::MakeDecision(AudioEncoderRuntimeConfig* config) {
  // Decision on `num_channels` should not have been made.
  RTC_DCHECK(!config->num_channels);

  if (uplink_bandwidth_bps_) {
    if (channels_to_encode_ == 2 &&
        *uplink_bandwidth_bps_ <= config_.channel_2_to_1_bandwidth_bps) {
      channels_to_encode_ = 1;
    } else if (channels_to_encode_ == 1 &&
               *uplink_bandwidth_bps_ >= config_.channel_1_to_2_bandwidth_bps) {
      channels_to_encode_ =
          std::min(static_cast<size_t>(2), config_.num_encoder_channels);
    }
  }
  config->num_channels = channels_to_encode_;
}

}  // namespace webrtc
