/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
#include "modules/audio_coding/audio_network_adaptor/frame_length_controller_v2.h"

#include <algorithm>
#include <memory>

#include "modules/audio_coding/audio_network_adaptor/controller.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor_config.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

constexpr int kANASupportedFrameLengths[] = {20, 40, 60, 120};
constexpr int kMinPayloadBitrateBps = 16000;

}  // namespace

class FrameLengthControllerV2Test : public testing::Test {
 protected:
  AudioEncoderRuntimeConfig GetDecision() {
    AudioEncoderRuntimeConfig config;
    controller_->MakeDecision(&config);
    return config;
  }

  void SetOverhead(int overhead_bytes_per_packet) {
    overhead_bytes_per_packet_ = overhead_bytes_per_packet;
    Controller::NetworkMetrics metrics;
    metrics.overhead_bytes_per_packet = overhead_bytes_per_packet;
    controller_->UpdateNetworkMetrics(metrics);
  }

  void SetTargetBitrate(int target_audio_bitrate_bps) {
    target_audio_bitrate_bps_ = target_audio_bitrate_bps;
    Controller::NetworkMetrics metrics;
    metrics.target_audio_bitrate_bps = target_audio_bitrate_bps;
    controller_->UpdateNetworkMetrics(metrics);
  }

  void SetUplinkBandwidth(int uplink_bandwidth_bps) {
    Controller::NetworkMetrics metrics;
    metrics.uplink_bandwidth_bps = uplink_bandwidth_bps;
    controller_->UpdateNetworkMetrics(metrics);
  }

  void ExpectFrameLengthDecision(int expected_frame_length_ms) {
    auto config = GetDecision();
    EXPECT_EQ(*config.frame_length_ms, expected_frame_length_ms);
  }

  std::unique_ptr<FrameLengthControllerV2> controller_ =
      std::make_unique<FrameLengthControllerV2>(kANASupportedFrameLengths,
                                                kMinPayloadBitrateBps,
                                                /*use_slow_adaptation=*/false);
  std::optional<int> target_audio_bitrate_bps_;
  std::optional<int> overhead_bytes_per_packet_;
};

// Don't return any decision if we haven't received all required network
// metrics.
TEST_F(FrameLengthControllerV2Test, RequireNetworkMetrics) {
  auto config = GetDecision();
  EXPECT_FALSE(config.bitrate_bps);
  EXPECT_FALSE(config.frame_length_ms);

  SetOverhead(30);
  config = GetDecision();
  EXPECT_FALSE(config.frame_length_ms);

  SetTargetBitrate(32000);
  config = GetDecision();
  EXPECT_FALSE(config.frame_length_ms);

  SetUplinkBandwidth(32000);
  config = GetDecision();
  EXPECT_TRUE(config.frame_length_ms);
}

TEST_F(FrameLengthControllerV2Test, UseFastAdaptation) {
  SetOverhead(50);
  SetTargetBitrate(50000);
  SetUplinkBandwidth(50000);
  ExpectFrameLengthDecision(20);

  SetTargetBitrate(20000);
  ExpectFrameLengthDecision(120);

  SetTargetBitrate(30000);
  ExpectFrameLengthDecision(40);

  SetTargetBitrate(25000);
  ExpectFrameLengthDecision(60);
}

TEST_F(FrameLengthControllerV2Test, UseSlowAdaptation) {
  controller_ = std::make_unique<FrameLengthControllerV2>(
      kANASupportedFrameLengths, kMinPayloadBitrateBps,
      /*use_slow_adaptation=*/true);
  SetOverhead(50);
  SetTargetBitrate(50000);
  SetUplinkBandwidth(20000);
  ExpectFrameLengthDecision(120);

  SetUplinkBandwidth(30000);
  ExpectFrameLengthDecision(40);

  SetUplinkBandwidth(40000);
  ExpectFrameLengthDecision(20);
}

}  // namespace webrtc
