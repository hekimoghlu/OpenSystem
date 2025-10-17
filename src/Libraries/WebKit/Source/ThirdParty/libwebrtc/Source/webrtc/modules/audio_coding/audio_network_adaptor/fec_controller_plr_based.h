/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_FEC_CONTROLLER_PLR_BASED_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_FEC_CONTROLLER_PLR_BASED_H_

#include <memory>
#include <optional>

#include "common_audio/smoothing_filter.h"
#include "modules/audio_coding/audio_network_adaptor/controller.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor_config.h"
#include "modules/audio_coding/audio_network_adaptor/util/threshold_curve.h"

namespace webrtc {

class FecControllerPlrBased final : public Controller {
 public:
  struct Config {
    // `fec_enabling_threshold` defines a curve, above which FEC should be
    // enabled. `fec_disabling_threshold` defines a curve, under which FEC
    // should be disabled. See below
    //
    // packet-loss ^   |  |
    //             |   |  |   FEC
    //             |    \  \   ON
    //             | FEC \  \_______ fec_enabling_threshold
    //             | OFF  \_________ fec_disabling_threshold
    //             |-----------------> bandwidth
    Config(bool initial_fec_enabled,
           const ThresholdCurve& fec_enabling_threshold,
           const ThresholdCurve& fec_disabling_threshold,
           int time_constant_ms);
    bool initial_fec_enabled;
    ThresholdCurve fec_enabling_threshold;
    ThresholdCurve fec_disabling_threshold;
    int time_constant_ms;
  };

  // Dependency injection for testing.
  FecControllerPlrBased(const Config& config,
                        std::unique_ptr<SmoothingFilter> smoothing_filter);

  explicit FecControllerPlrBased(const Config& config);

  ~FecControllerPlrBased() override;

  FecControllerPlrBased(const FecControllerPlrBased&) = delete;
  FecControllerPlrBased& operator=(const FecControllerPlrBased&) = delete;

  void UpdateNetworkMetrics(const NetworkMetrics& network_metrics) override;

  void MakeDecision(AudioEncoderRuntimeConfig* config) override;

 private:
  bool FecEnablingDecision(const std::optional<float>& packet_loss) const;
  bool FecDisablingDecision(const std::optional<float>& packet_loss) const;

  const Config config_;
  bool fec_enabled_;
  std::optional<int> uplink_bandwidth_bps_;
  const std::unique_ptr<SmoothingFilter> packet_loss_smoother_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_FEC_CONTROLLER_PLR_BASED_H_
