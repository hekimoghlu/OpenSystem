/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_BITRATE_CONTROLLER_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_BITRATE_CONTROLLER_H_

#include <stddef.h>

#include <optional>

#include "modules/audio_coding/audio_network_adaptor/controller.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor_config.h"

namespace webrtc {
namespace audio_network_adaptor {

class BitrateController final : public Controller {
 public:
  struct Config {
    Config(int initial_bitrate_bps,
           int initial_frame_length_ms,
           int fl_increase_overhead_offset,
           int fl_decrease_overhead_offset);
    ~Config();
    int initial_bitrate_bps;
    int initial_frame_length_ms;
    int fl_increase_overhead_offset;
    int fl_decrease_overhead_offset;
  };

  explicit BitrateController(const Config& config);

  ~BitrateController() override;

  BitrateController(const BitrateController&) = delete;
  BitrateController& operator=(const BitrateController&) = delete;

  void UpdateNetworkMetrics(const NetworkMetrics& network_metrics) override;

  void MakeDecision(AudioEncoderRuntimeConfig* config) override;

 private:
  const Config config_;
  int bitrate_bps_;
  int frame_length_ms_;
  std::optional<int> target_audio_bitrate_bps_;
  std::optional<size_t> overhead_bytes_per_packet_;
};

}  // namespace audio_network_adaptor
}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_BITRATE_CONTROLLER_H_
