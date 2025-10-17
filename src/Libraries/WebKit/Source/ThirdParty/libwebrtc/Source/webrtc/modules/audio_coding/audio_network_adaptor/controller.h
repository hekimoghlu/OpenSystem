/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CONTROLLER_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CONTROLLER_H_

#include <optional>

#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor.h"

namespace webrtc {

class Controller {
 public:
  struct NetworkMetrics {
    NetworkMetrics();
    ~NetworkMetrics();
    std::optional<int> uplink_bandwidth_bps;
    std::optional<float> uplink_packet_loss_fraction;
    std::optional<int> target_audio_bitrate_bps;
    std::optional<int> rtt_ms;
    std::optional<size_t> overhead_bytes_per_packet;
  };

  virtual ~Controller() = default;

  // Informs network metrics update to this controller. Any non-empty field
  // indicates an update on the corresponding network metric.
  virtual void UpdateNetworkMetrics(const NetworkMetrics& network_metrics) = 0;

  virtual void MakeDecision(AudioEncoderRuntimeConfig* config) = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CONTROLLER_H_
