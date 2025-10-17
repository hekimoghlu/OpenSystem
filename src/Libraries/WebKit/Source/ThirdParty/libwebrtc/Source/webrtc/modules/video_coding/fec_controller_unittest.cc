/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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
#include "api/fec_controller.h"

#include <stdint.h>

#include <vector>

#include "api/environment/environment_factory.h"
#include "modules/include/module_fec_types.h"
#include "modules/video_coding/fec_controller_default.h"
#include "system_wrappers/include/clock.h"
#include "test/gtest.h"

namespace webrtc {

static const int kCodecBitrateBps = 100000;

class ProtectionBitrateCalculatorTest : public ::testing::Test {
 protected:
  enum {
    kSampleRate = 90000  // RTP timestamps per second.
  };

  class ProtectionCallback : public VCMProtectionCallback {
   public:
    int ProtectionRequest(const FecProtectionParams* delta_params,
                          const FecProtectionParams* key_params,
                          uint32_t* sent_video_rate_bps,
                          uint32_t* sent_nack_rate_bps,
                          uint32_t* sent_fec_rate_bps) override {
      *sent_video_rate_bps = kCodecBitrateBps;
      *sent_nack_rate_bps = nack_rate_bps_;
      *sent_fec_rate_bps = fec_rate_bps_;
      return 0;
    }
    void SetRetransmissionMode(int retransmission_mode) {}

    uint32_t fec_rate_bps_ = 0;
    uint32_t nack_rate_bps_ = 0;
  };

  // Note: simulated clock starts at 1 seconds, since parts of webrtc use 0 as
  // a special case (e.g. frame rate in media optimization).
  ProtectionBitrateCalculatorTest()
      : clock_(1000),
        fec_controller_(CreateEnvironment(&clock_), &protection_callback_) {}

  SimulatedClock clock_;
  ProtectionCallback protection_callback_;
  FecControllerDefault fec_controller_;
};

TEST_F(ProtectionBitrateCalculatorTest, ProtectsUsingFecBitrate) {
  static const uint32_t kMaxBitrateBps = 130000;

  fec_controller_.SetProtectionMethod(true /*enable_fec*/,
                                      false /* enable_nack */);
  fec_controller_.SetEncodingData(640, 480, 1, 1000);

  // Using 10% of codec bitrate for FEC.
  protection_callback_.fec_rate_bps_ = kCodecBitrateBps / 10;
  uint32_t target_bitrate = fec_controller_.UpdateFecRates(
      kMaxBitrateBps, 30, 0, std::vector<bool>(1, false), 0);

  EXPECT_GT(target_bitrate, 0u);
  EXPECT_GT(kMaxBitrateBps, target_bitrate);

  // Using as much for codec bitrate as fec rate, new target rate should share
  // both equally, but only be half of max (since that ceiling should be hit).
  protection_callback_.fec_rate_bps_ = kCodecBitrateBps;
  target_bitrate = fec_controller_.UpdateFecRates(
      kMaxBitrateBps, 30, 128, std::vector<bool>(1, false), 100);
  EXPECT_EQ(kMaxBitrateBps / 2, target_bitrate);
}

TEST_F(ProtectionBitrateCalculatorTest, ProtectsUsingNackBitrate) {
  static const uint32_t kMaxBitrateBps = 130000;

  fec_controller_.SetProtectionMethod(false /*enable_fec*/,
                                      true /* enable_nack */);
  fec_controller_.SetEncodingData(640, 480, 1, 1000);

  uint32_t target_bitrate = fec_controller_.UpdateFecRates(
      kMaxBitrateBps, 30, 0, std::vector<bool>(1, false), 0);

  EXPECT_EQ(kMaxBitrateBps, target_bitrate);

  // Using as much for codec bitrate as nack rate, new target rate should share
  // both equally, but only be half of max (since that ceiling should be hit).
  protection_callback_.nack_rate_bps_ = kMaxBitrateBps;
  target_bitrate = fec_controller_.UpdateFecRates(
      kMaxBitrateBps, 30, 128, std::vector<bool>(1, false), 100);
  EXPECT_EQ(kMaxBitrateBps / 2, target_bitrate);
}

TEST_F(ProtectionBitrateCalculatorTest, NoProtection) {
  static const uint32_t kMaxBitrateBps = 130000;

  fec_controller_.SetProtectionMethod(false /*enable_fec*/,
                                      false /* enable_nack */);
  fec_controller_.SetEncodingData(640, 480, 1, 1000);

  uint32_t target_bitrate = fec_controller_.UpdateFecRates(
      kMaxBitrateBps, 30, 128, std::vector<bool>(1, false), 100);
  EXPECT_EQ(kMaxBitrateBps, target_bitrate);
}

}  // namespace webrtc
