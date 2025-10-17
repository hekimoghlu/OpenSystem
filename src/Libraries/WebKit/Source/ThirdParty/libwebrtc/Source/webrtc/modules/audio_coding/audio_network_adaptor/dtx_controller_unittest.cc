/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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

#include <memory>

#include "test/gtest.h"

namespace webrtc {

namespace {

constexpr int kDtxEnablingBandwidthBps = 55000;
constexpr int kDtxDisablingBandwidthBps = 65000;
constexpr int kMediumBandwidthBps =
    (kDtxEnablingBandwidthBps + kDtxDisablingBandwidthBps) / 2;

std::unique_ptr<DtxController> CreateController(int initial_dtx_enabled) {
  std::unique_ptr<DtxController> controller(new DtxController(
      DtxController::Config(initial_dtx_enabled, kDtxEnablingBandwidthBps,
                            kDtxDisablingBandwidthBps)));
  return controller;
}

void CheckDecision(DtxController* controller,
                   const std::optional<int>& uplink_bandwidth_bps,
                   bool expected_dtx_enabled) {
  if (uplink_bandwidth_bps) {
    Controller::NetworkMetrics network_metrics;
    network_metrics.uplink_bandwidth_bps = uplink_bandwidth_bps;
    controller->UpdateNetworkMetrics(network_metrics);
  }
  AudioEncoderRuntimeConfig config;
  controller->MakeDecision(&config);
  EXPECT_EQ(expected_dtx_enabled, config.enable_dtx);
}

}  // namespace

TEST(DtxControllerTest, OutputInitValueWhenUplinkBandwidthUnknown) {
  constexpr bool kInitialDtxEnabled = true;
  auto controller = CreateController(kInitialDtxEnabled);
  CheckDecision(controller.get(), std::nullopt, kInitialDtxEnabled);
}

TEST(DtxControllerTest, TurnOnDtxForLowUplinkBandwidth) {
  auto controller = CreateController(false);
  CheckDecision(controller.get(), kDtxEnablingBandwidthBps, true);
}

TEST(DtxControllerTest, TurnOffDtxForHighUplinkBandwidth) {
  auto controller = CreateController(true);
  CheckDecision(controller.get(), kDtxDisablingBandwidthBps, false);
}

TEST(DtxControllerTest, MaintainDtxOffForMediumUplinkBandwidth) {
  auto controller = CreateController(false);
  CheckDecision(controller.get(), kMediumBandwidthBps, false);
}

TEST(DtxControllerTest, MaintainDtxOnForMediumUplinkBandwidth) {
  auto controller = CreateController(true);
  CheckDecision(controller.get(), kMediumBandwidthBps, true);
}

TEST(DtxControllerTest, CheckBehaviorOnChangingUplinkBandwidth) {
  auto controller = CreateController(false);
  CheckDecision(controller.get(), kMediumBandwidthBps, false);
  CheckDecision(controller.get(), kDtxEnablingBandwidthBps, true);
  CheckDecision(controller.get(), kMediumBandwidthBps, true);
  CheckDecision(controller.get(), kDtxDisablingBandwidthBps, false);
}

}  // namespace webrtc
