/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#include "modules/audio_coding/audio_network_adaptor/fec_controller_plr_based.h"

#include <string>
#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {

FecControllerPlrBased::Config::Config(
    bool initial_fec_enabled,
    const ThresholdCurve& fec_enabling_threshold,
    const ThresholdCurve& fec_disabling_threshold,
    int time_constant_ms)
    : initial_fec_enabled(initial_fec_enabled),
      fec_enabling_threshold(fec_enabling_threshold),
      fec_disabling_threshold(fec_disabling_threshold),
      time_constant_ms(time_constant_ms) {}

FecControllerPlrBased::FecControllerPlrBased(
    const Config& config,
    std::unique_ptr<SmoothingFilter> smoothing_filter)
    : config_(config),
      fec_enabled_(config.initial_fec_enabled),
      packet_loss_smoother_(std::move(smoothing_filter)) {
  RTC_DCHECK(config_.fec_disabling_threshold <= config_.fec_enabling_threshold);
}

FecControllerPlrBased::FecControllerPlrBased(const Config& config)
    : FecControllerPlrBased(
          config,
          std::make_unique<SmoothingFilterImpl>(config.time_constant_ms)) {}

FecControllerPlrBased::~FecControllerPlrBased() = default;

void FecControllerPlrBased::UpdateNetworkMetrics(
    const NetworkMetrics& network_metrics) {
  if (network_metrics.uplink_bandwidth_bps)
    uplink_bandwidth_bps_ = network_metrics.uplink_bandwidth_bps;
  if (network_metrics.uplink_packet_loss_fraction) {
    packet_loss_smoother_->AddSample(
        *network_metrics.uplink_packet_loss_fraction);
  }
}

void FecControllerPlrBased::MakeDecision(AudioEncoderRuntimeConfig* config) {
  RTC_DCHECK(!config->enable_fec);
  RTC_DCHECK(!config->uplink_packet_loss_fraction);

  const auto& packet_loss = packet_loss_smoother_->GetAverage();

  fec_enabled_ = fec_enabled_ ? !FecDisablingDecision(packet_loss)
                              : FecEnablingDecision(packet_loss);

  config->enable_fec = fec_enabled_;

  config->uplink_packet_loss_fraction = packet_loss ? *packet_loss : 0.0;
}

bool FecControllerPlrBased::FecEnablingDecision(
    const std::optional<float>& packet_loss) const {
  if (!uplink_bandwidth_bps_ || !packet_loss) {
    return false;
  } else {
    // Enable when above the curve or exactly on it.
    return !config_.fec_enabling_threshold.IsBelowCurve(
        {static_cast<float>(*uplink_bandwidth_bps_), *packet_loss});
  }
}

bool FecControllerPlrBased::FecDisablingDecision(
    const std::optional<float>& packet_loss) const {
  if (!uplink_bandwidth_bps_ || !packet_loss) {
    return false;
  } else {
    // Disable when below the curve.
    return config_.fec_disabling_threshold.IsBelowCurve(
        {static_cast<float>(*uplink_bandwidth_bps_), *packet_loss});
  }
}

}  // namespace webrtc
