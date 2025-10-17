/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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
#include <memory>

#include "absl/flags/flag.h"
#include "modules/audio_coding/codecs/pcm16b/audio_encoder_pcm16b.h"
#include "modules/audio_coding/neteq/tools/neteq_quality_test.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "test/testsupport/file_utils.h"

ABSL_FLAG(int, frame_size_ms, 20, "Codec frame size (milliseconds).");

using ::testing::InitGoogleTest;

namespace webrtc {
namespace test {
namespace {
static const int kInputSampleRateKhz = 48;
static const int kOutputSampleRateKhz = 48;
}  // namespace

class NetEqPcm16bQualityTest : public NetEqQualityTest {
 protected:
  NetEqPcm16bQualityTest()
      : NetEqQualityTest(absl::GetFlag(FLAGS_frame_size_ms),
                         kInputSampleRateKhz,
                         kOutputSampleRateKhz,
                         SdpAudioFormat("l16", 48000, 1)) {
    // Flag validation
    RTC_CHECK(absl::GetFlag(FLAGS_frame_size_ms) >= 10 &&
              absl::GetFlag(FLAGS_frame_size_ms) <= 60 &&
              (absl::GetFlag(FLAGS_frame_size_ms) % 10) == 0)
        << "Invalid frame size, should be 10, 20, ..., 60 ms.";
  }

  void SetUp() override {
    AudioEncoderPcm16B::Config config;
    config.frame_size_ms = absl::GetFlag(FLAGS_frame_size_ms);
    config.sample_rate_hz = 48000;
    config.num_channels = channels_;
    encoder_.reset(new AudioEncoderPcm16B(config));
    NetEqQualityTest::SetUp();
  }

  int EncodeBlock(int16_t* in_data,
                  size_t /* block_size_samples */,
                  rtc::Buffer* payload,
                  size_t /* max_bytes */) override {
    const size_t kFrameSizeSamples = 480;  // Samples per 10 ms.
    size_t encoded_samples = 0;
    uint32_t dummy_timestamp = 0;
    AudioEncoder::EncodedInfo info;
    do {
      info = encoder_->Encode(dummy_timestamp,
                              rtc::ArrayView<const int16_t>(
                                  in_data + encoded_samples, kFrameSizeSamples),
                              payload);
      encoded_samples += kFrameSizeSamples;
    } while (info.encoded_bytes == 0);
    return rtc::checked_cast<int>(info.encoded_bytes);
  }

 private:
  std::unique_ptr<AudioEncoderPcm16B> encoder_;
};

TEST_F(NetEqPcm16bQualityTest, Test) {
  Simulate();
}

}  // namespace test
}  // namespace webrtc
