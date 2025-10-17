/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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
#include "modules/audio_coding/audio_network_adaptor/dtx_controller.h"

#include "rtc_base/checks.h"

namespace webrtc {

DtxController::Config::Config(bool initial_dtx_enabled,
                              int dtx_enabling_bandwidth_bps,
                              int dtx_disabling_bandwidth_bps)
    : initial_dtx_enabled(initial_dtx_enabled),
      dtx_enabling_bandwidth_bps(dtx_enabling_bandwidth_bps),
      dtx_disabling_bandwidth_bps(dtx_disabling_bandwidth_bps) {}

DtxController::DtxController(const Config& config)
    : config_(config), dtx_enabled_(config_.initial_dtx_enabled) {}

DtxController::~DtxController() = default;

void DtxController::UpdateNetworkMetrics(
    const NetworkMetrics& network_metrics) {
  if (network_metrics.uplink_bandwidth_bps)
    uplink_bandwidth_bps_ = network_metrics.uplink_bandwidth_bps;
}

void DtxController::MakeDecision(AudioEncoderRuntimeConfig* config) {
  // Decision on `enable_dtx` should not have been made.
  RTC_DCHECK(!config->enable_dtx);

  if (uplink_bandwidth_bps_) {
    if (dtx_enabled_ &&
        *uplink_bandwidth_bps_ >= config_.dtx_disabling_bandwidth_bps) {
      dtx_enabled_ = false;
    } else if (!dtx_enabled_ &&
               *uplink_bandwidth_bps_ <= config_.dtx_enabling_bandwidth_bps) {
      dtx_enabled_ = true;
    }
  }
  config->enable_dtx = dtx_enabled_;
}

}  // namespace webrtc
