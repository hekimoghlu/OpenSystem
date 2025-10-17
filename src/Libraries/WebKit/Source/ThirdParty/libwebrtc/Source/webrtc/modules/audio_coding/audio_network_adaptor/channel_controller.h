/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CHANNEL_CONTROLLER_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CHANNEL_CONTROLLER_H_

#include <stddef.h>

#include <optional>

#include "modules/audio_coding/audio_network_adaptor/controller.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor_config.h"

namespace webrtc {

class ChannelController final : public Controller {
 public:
  struct Config {
    Config(size_t num_encoder_channels,
           size_t intial_channels_to_encode,
           int channel_1_to_2_bandwidth_bps,
           int channel_2_to_1_bandwidth_bps);
    size_t num_encoder_channels;
    size_t intial_channels_to_encode;
    // Uplink bandwidth above which the number of encoded channels should switch
    // from 1 to 2.
    int channel_1_to_2_bandwidth_bps;
    // Uplink bandwidth below which the number of encoded channels should switch
    // from 2 to 1.
    int channel_2_to_1_bandwidth_bps;
  };

  explicit ChannelController(const Config& config);

  ~ChannelController() override;

  ChannelController(const ChannelController&) = delete;
  ChannelController& operator=(const ChannelController&) = delete;

  void UpdateNetworkMetrics(const NetworkMetrics& network_metrics) override;

  void MakeDecision(AudioEncoderRuntimeConfig* config) override;

 private:
  const Config config_;
  size_t channels_to_encode_;
  std::optional<int> uplink_bandwidth_bps_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CHANNEL_CONTROLLER_H_
