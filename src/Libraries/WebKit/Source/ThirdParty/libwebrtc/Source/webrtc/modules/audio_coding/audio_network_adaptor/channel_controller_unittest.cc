/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

#include <memory>

#include "test/gtest.h"

namespace webrtc {

namespace {

constexpr int kNumChannels = 2;
constexpr int kChannel1To2BandwidthBps = 31000;
constexpr int kChannel2To1BandwidthBps = 29000;
constexpr int kMediumBandwidthBps =
    (kChannel1To2BandwidthBps + kChannel2To1BandwidthBps) / 2;

std::unique_ptr<ChannelController> CreateChannelController(int init_channels) {
  std::unique_ptr<ChannelController> controller(
      new ChannelController(ChannelController::Config(
          kNumChannels, init_channels, kChannel1To2BandwidthBps,
          kChannel2To1BandwidthBps)));
  return controller;
}

void CheckDecision(ChannelController* controller,
                   const std::optional<int>& uplink_bandwidth_bps,
                   size_t expected_num_channels) {
  if (uplink_bandwidth_bps) {
    Controller::NetworkMetrics network_metrics;
    network_metrics.uplink_bandwidth_bps = uplink_bandwidth_bps;
    controller->UpdateNetworkMetrics(network_metrics);
  }
  AudioEncoderRuntimeConfig config;
  controller->MakeDecision(&config);
  EXPECT_EQ(expected_num_channels, config.num_channels);
}

}  // namespace

TEST(ChannelControllerTest, OutputInitValueWhenUplinkBandwidthUnknown) {
  constexpr int kInitChannels = 2;
  auto controller = CreateChannelController(kInitChannels);
  CheckDecision(controller.get(), std::nullopt, kInitChannels);
}

TEST(ChannelControllerTest, SwitchTo2ChannelsOnHighUplinkBandwidth) {
  constexpr int kInitChannels = 1;
  auto controller = CreateChannelController(kInitChannels);
  // Use high bandwidth to check output switch to 2.
  CheckDecision(controller.get(), kChannel1To2BandwidthBps, 2);
}

TEST(ChannelControllerTest, SwitchTo1ChannelOnLowUplinkBandwidth) {
  constexpr int kInitChannels = 2;
  auto controller = CreateChannelController(kInitChannels);
  // Use low bandwidth to check output switch to 1.
  CheckDecision(controller.get(), kChannel2To1BandwidthBps, 1);
}

TEST(ChannelControllerTest, Maintain1ChannelOnMediumUplinkBandwidth) {
  constexpr int kInitChannels = 1;
  auto controller = CreateChannelController(kInitChannels);
  // Use between-thresholds bandwidth to check output remains at 1.
  CheckDecision(controller.get(), kMediumBandwidthBps, 1);
}

TEST(ChannelControllerTest, Maintain2ChannelsOnMediumUplinkBandwidth) {
  constexpr int kInitChannels = 2;
  auto controller = CreateChannelController(kInitChannels);
  // Use between-thresholds bandwidth to check output remains at 2.
  CheckDecision(controller.get(), kMediumBandwidthBps, 2);
}

TEST(ChannelControllerTest, CheckBehaviorOnChangingUplinkBandwidth) {
  constexpr int kInitChannels = 1;
  auto controller = CreateChannelController(kInitChannels);

  // Use between-thresholds bandwidth to check output remains at 1.
  CheckDecision(controller.get(), kMediumBandwidthBps, 1);

  // Use high bandwidth to check output switch to 2.
  CheckDecision(controller.get(), kChannel1To2BandwidthBps, 2);

  // Use between-thresholds bandwidth to check output remains at 2.
  CheckDecision(controller.get(), kMediumBandwidthBps, 2);

  // Use low bandwidth to check output switch to 1.
  CheckDecision(controller.get(), kChannel2To1BandwidthBps, 1);
}

}  // namespace webrtc
