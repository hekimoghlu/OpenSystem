/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#include "test/gtest.h"
#include "test/scenario/scenario.h"

namespace webrtc {
namespace test {

TEST(ProbingTest, InitialProbingRampsUpTargetRateWhenNetworkIsGood) {
  Scenario s;
  NetworkSimulationConfig good_network;
  good_network.bandwidth = DataRate::KilobitsPerSec(2000);

  VideoStreamConfig video_config;
  video_config.encoder.codec =
      VideoStreamConfig::Encoder::Codec::kVideoCodecVP8;
  CallClientConfig send_config;
  auto* caller = s.CreateClient("caller", send_config);
  auto* callee = s.CreateClient("callee", CallClientConfig());
  auto route =
      s.CreateRoutes(caller, {s.CreateSimulationNode(good_network)}, callee,
                     {s.CreateSimulationNode(NetworkSimulationConfig())});
  s.CreateVideoStream(route->forward(), video_config);

  s.RunFor(TimeDelta::Seconds(1));
  EXPECT_GE(DataRate::BitsPerSec(caller->GetStats().send_bandwidth_bps),
            3 * send_config.transport.rates.start_rate);
}

TEST(ProbingTest, MidCallProbingRampupTriggeredByUpdatedBitrateConstraints) {
  Scenario s;

  const DataRate kStartRate = DataRate::KilobitsPerSec(300);
  const DataRate kConstrainedRate = DataRate::KilobitsPerSec(100);
  const DataRate kHighRate = DataRate::KilobitsPerSec(1500);

  VideoStreamConfig video_config;
  video_config.encoder.codec =
      VideoStreamConfig::Encoder::Codec::kVideoCodecVP8;
  CallClientConfig send_call_config;
  send_call_config.transport.rates.start_rate = kStartRate;
  send_call_config.transport.rates.max_rate = kHighRate * 2;
  auto* caller = s.CreateClient("caller", send_call_config);
  auto* callee = s.CreateClient("callee", CallClientConfig());
  auto route = s.CreateRoutes(
      caller, {s.CreateSimulationNode(NetworkSimulationConfig())}, callee,
      {s.CreateSimulationNode(NetworkSimulationConfig())});
  s.CreateVideoStream(route->forward(), video_config);

  // Wait until initial probing rampup is done and then set a low max bitrate.
  s.RunFor(TimeDelta::Seconds(1));
  EXPECT_GE(DataRate::BitsPerSec(caller->GetStats().send_bandwidth_bps),
            5 * send_call_config.transport.rates.start_rate);
  BitrateConstraints bitrate_config;
  bitrate_config.max_bitrate_bps = kConstrainedRate.bps();
  caller->UpdateBitrateConstraints(bitrate_config);

  // Wait until the low send bitrate has taken effect, and then set a much
  // higher max bitrate.
  s.RunFor(TimeDelta::Seconds(2));
  EXPECT_LT(DataRate::BitsPerSec(caller->GetStats().send_bandwidth_bps),
            kConstrainedRate * 1.1);
  bitrate_config.max_bitrate_bps = 2 * kHighRate.bps();
  caller->UpdateBitrateConstraints(bitrate_config);

  // Check that the max send bitrate is reached quicker than would be possible
  // with simple AIMD rate control.
  s.RunFor(TimeDelta::Seconds(1));
  EXPECT_GE(DataRate::BitsPerSec(caller->GetStats().send_bandwidth_bps),
            kHighRate);
}

TEST(ProbingTest, ProbesRampsUpWhenVideoEncoderConfigChanges) {
  Scenario s;
  const DataRate kStartRate = DataRate::KilobitsPerSec(50);
  const DataRate kHdRate = DataRate::KilobitsPerSec(3250);

  // Set up 3-layer simulcast.
  VideoStreamConfig video_config;
  video_config.encoder.codec =
      VideoStreamConfig::Encoder::Codec::kVideoCodecVP8;
  video_config.encoder.simulcast_streams = {webrtc::ScalabilityMode::kL1T3,
                                            webrtc::ScalabilityMode::kL1T3,
                                            webrtc::ScalabilityMode::kL1T3};
  video_config.source.generator.width = 1280;
  video_config.source.generator.height = 720;

  CallClientConfig send_call_config;
  send_call_config.transport.rates.start_rate = kStartRate;
  send_call_config.transport.rates.max_rate = kHdRate * 2;
  auto* caller = s.CreateClient("caller", send_call_config);
  auto* callee = s.CreateClient("callee", CallClientConfig());
  auto send_net =
      s.CreateMutableSimulationNode([&](NetworkSimulationConfig* c) {
        c->bandwidth = DataRate::KilobitsPerSec(200);
      });
  auto route =
      s.CreateRoutes(caller, {send_net->node()}, callee,
                     {s.CreateSimulationNode(NetworkSimulationConfig())});
  auto* video_stream = s.CreateVideoStream(route->forward(), video_config);

  // Only QVGA enabled initially. Run until initial probing is done and BWE
  // has settled.
  video_stream->send()->UpdateActiveLayers({true, false, false});
  s.RunFor(TimeDelta::Seconds(2));

  // Remove network constraints and run for a while more, BWE should be much
  // less than required HD rate.
  send_net->UpdateConfig([&](NetworkSimulationConfig* c) {
    c->bandwidth = DataRate::PlusInfinity();
  });
  s.RunFor(TimeDelta::Seconds(2));

  DataRate bandwidth =
      DataRate::BitsPerSec(caller->GetStats().send_bandwidth_bps);
  EXPECT_LT(bandwidth, kHdRate / 4);

  // Enable all layers, triggering a probe.
  video_stream->send()->UpdateActiveLayers({true, true, true});

  // Run for a short while and verify BWE has ramped up fast.
  s.RunFor(TimeDelta::Seconds(2));
  EXPECT_GT(DataRate::BitsPerSec(caller->GetStats().send_bandwidth_bps),
            kHdRate);
}

}  // namespace test
}  // namespace webrtc
