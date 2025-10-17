/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#include "modules/audio_coding/codecs/g711/audio_encoder_pcm.h"
#include "modules/audio_coding/neteq/tools/neteq_quality_test.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "test/testsupport/file_utils.h"

ABSL_FLAG(int, frame_size_ms, 20, "Codec frame size (milliseconds).");

using ::testing::InitGoogleTest;

namespace webrtc {
namespace test {
namespace {
static const int kInputSampleRateKhz = 8;
static const int kOutputSampleRateKhz = 8;
}  // namespace

class NetEqPcmuQualityTest : public NetEqQualityTest {
 protected:
  NetEqPcmuQualityTest()
      : NetEqQualityTest(absl::GetFlag(FLAGS_frame_size_ms),
                         kInputSampleRateKhz,
                         kOutputSampleRateKhz,
                         SdpAudioFormat("pcmu", 8000, 1)) {
    // Flag validation
    RTC_CHECK(absl::GetFlag(FLAGS_frame_size_ms) >= 10 &&
              absl::GetFlag(FLAGS_frame_size_ms) <= 60 &&
              (absl::GetFlag(FLAGS_frame_size_ms) % 10) == 0)
        << "Invalid frame size, should be 10, 20, ..., 60 ms.";
  }

  void SetUp() override {
    ASSERT_EQ(1u, channels_) << "PCMu supports only mono audio.";
    AudioEncoderPcmU::Config config;
    config.frame_size_ms = absl::GetFlag(FLAGS_frame_size_ms);
    encoder_.reset(new AudioEncoderPcmU(config));
    NetEqQualityTest::SetUp();
  }

  int EncodeBlock(int16_t* in_data,
                  size_t /* block_size_samples */,
                  rtc::Buffer* payload,
                  size_t /* max_bytes */) override {
    const size_t kFrameSizeSamples = 80;  // Samples per 10 ms.
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
  std::unique_ptr<AudioEncoderPcmU> encoder_;
};

TEST_F(NetEqPcmuQualityTest, Test) {
  Simulate();
}

}  // namespace test
}  // namespace webrtc
