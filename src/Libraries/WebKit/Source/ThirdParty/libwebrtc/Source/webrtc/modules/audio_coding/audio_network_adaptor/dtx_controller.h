/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_DTX_CONTROLLER_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_DTX_CONTROLLER_H_

#include <optional>

#include "modules/audio_coding/audio_network_adaptor/controller.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor_config.h"

namespace webrtc {

class DtxController final : public Controller {
 public:
  struct Config {
    Config(bool initial_dtx_enabled,
           int dtx_enabling_bandwidth_bps,
           int dtx_disabling_bandwidth_bps);
    bool initial_dtx_enabled;
    // Uplink bandwidth below which DTX should be switched on.
    int dtx_enabling_bandwidth_bps;
    // Uplink bandwidth above which DTX should be switched off.
    int dtx_disabling_bandwidth_bps;
  };

  explicit DtxController(const Config& config);

  ~DtxController() override;

  DtxController(const DtxController&) = delete;
  DtxController& operator=(const DtxController&) = delete;

  void UpdateNetworkMetrics(const NetworkMetrics& network_metrics) override;

  void MakeDecision(AudioEncoderRuntimeConfig* config) override;

 private:
  const Config config_;
  bool dtx_enabled_;
  std::optional<int> uplink_bandwidth_bps_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_DTX_CONTROLLER_H_
