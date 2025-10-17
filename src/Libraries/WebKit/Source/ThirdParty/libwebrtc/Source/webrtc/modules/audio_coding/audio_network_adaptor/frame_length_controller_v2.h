/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_FRAME_LENGTH_CONTROLLER_V2_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_FRAME_LENGTH_CONTROLLER_V2_H_

#include <optional>
#include <vector>

#include "modules/audio_coding/audio_network_adaptor/controller.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor.h"

namespace webrtc {

class FrameLengthControllerV2 final : public Controller {
 public:
  FrameLengthControllerV2(rtc::ArrayView<const int> encoder_frame_lengths_ms,
                          int min_payload_bitrate_bps,
                          bool use_slow_adaptation);

  void UpdateNetworkMetrics(const NetworkMetrics& network_metrics) override;

  void MakeDecision(AudioEncoderRuntimeConfig* config) override;

 private:
  std::vector<int> encoder_frame_lengths_ms_;
  const int min_payload_bitrate_bps_;
  const bool use_slow_adaptation_;

  std::optional<int> uplink_bandwidth_bps_;
  std::optional<int> target_bitrate_bps_;
  std::optional<int> overhead_bytes_per_packet_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_FRAME_LENGTH_CONTROLLER_V2_H_
