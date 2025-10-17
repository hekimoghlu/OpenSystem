/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
#include "api/audio_codecs/opus/audio_decoder_opus.h"
#include "api/audio_codecs/opus/audio_encoder_opus.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "common_audio/include/audio_util.h"
#include "common_audio/window_generator.h"
#include "modules/audio_coding/codecs/opus/test/lapped_transform.h"
#include "modules/audio_coding/neteq/tools/audio_loop.h"
#include "test/explicit_key_value_config.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace {

constexpr size_t kNumChannels = 1u;
constexpr int kSampleRateHz = 48000;
constexpr size_t kMaxLoopLengthSamples = kSampleRateHz * 50;  // 50 seconds.
constexpr size_t kInputBlockSizeSamples = 10 * kSampleRateHz / 1000;   // 10 ms
constexpr size_t kOutputBlockSizeSamples = 20 * kSampleRateHz / 1000;  // 20 ms
constexpr size_t kFftSize = 1024;
constexpr size_t kNarrowbandSize = 4000 * kFftSize / kSampleRateHz;
constexpr float kKbdAlpha = 1.5f;

class PowerRatioEstimator : public LappedTransform::Callback {
 public:
  PowerRatioEstimator() : low_pow_(0.f), high_pow_(0.f) {
    WindowGenerator::KaiserBesselDerived(kKbdAlpha, kFftSize, window_);
    transform_.reset(new LappedTransform(kNumChannels, 0u,
                                         kInputBlockSizeSamples, window_,
                                         kFftSize, kFftSize / 2, this));
  }

  void ProcessBlock(float* data) { transform_->ProcessChunk(&data, nullptr); }

  float PowerRatio() { return high_pow_ / low_pow_; }

 protected:
  void ProcessAudioBlock(const std::complex<float>* const* input,
                         size_t num_input_channels,
                         size_t /* num_freq_bins */,
                         size_t /* num_output_channels */,
                         std::complex<float>* const* /* output */) override {
    float low_pow = 0.f;
    float high_pow = 0.f;
    for (size_t i = 0u; i < num_input_channels; ++i) {
      for (size_t j = 0u; j < kNarrowbandSize; ++j) {
        float low_mag = std::abs(input[i][j]);
        low_pow += low_mag * low_mag;
        float high_mag = std::abs(input[i][j + kNarrowbandSize]);
        high_pow += high_mag * high_mag;
      }
    }
    low_pow_ += low_pow / (num_input_channels * kFftSize);
    high_pow_ += high_pow / (num_input_channels * kFftSize);
  }

 private:
  std::unique_ptr<LappedTransform> transform_;
  float window_[kFftSize];
  float low_pow_;
  float high_pow_;
};

float EncodedPowerRatio(AudioEncoder* encoder,
                        AudioDecoder* decoder,
                        test::AudioLoop* audio_loop) {
  // Encode and decode.
  uint32_t rtp_timestamp = 0u;
  constexpr size_t kBufferSize = 500;
  rtc::Buffer encoded(kBufferSize);
  std::vector<int16_t> decoded(kOutputBlockSizeSamples);
  std::vector<float> decoded_float(kOutputBlockSizeSamples);
  AudioDecoder::SpeechType speech_type = AudioDecoder::kSpeech;
  PowerRatioEstimator power_ratio_estimator;
  for (size_t i = 0; i < 1000; ++i) {
    encoded.Clear();
    AudioEncoder::EncodedInfo encoder_info =
        encoder->Encode(rtp_timestamp, audio_loop->GetNextBlock(), &encoded);
    rtp_timestamp += kInputBlockSizeSamples;
    if (encoded.size() > 0) {
      int decoder_info = decoder->Decode(
          encoded.data(), encoded.size(), kSampleRateHz,
          decoded.size() * sizeof(decoded[0]), decoded.data(), &speech_type);
      if (decoder_info > 0) {
        S16ToFloat(decoded.data(), decoded.size(), decoded_float.data());
        power_ratio_estimator.ProcessBlock(decoded_float.data());
      }
    }
  }
  return power_ratio_estimator.PowerRatio();
}

}  // namespace

// TODO(ivoc): Remove this test, WebRTC-AdjustOpusBandwidth is obsolete.
TEST(BandwidthAdaptationTest, BandwidthAdaptationTest) {
  const Environment env =
      CreateEnvironment(std::make_unique<test::ExplicitKeyValueConfig>(
          "WebRTC-AdjustOpusBandwidth/Enabled/"));

  constexpr float kMaxNarrowbandRatio = 0.0035f;
  constexpr float kMinWidebandRatio = 0.01f;

  // Create encoder.
  AudioEncoderOpusConfig enc_config;
  enc_config.bitrate_bps = std::optional<int>(7999);
  enc_config.num_channels = kNumChannels;
  auto encoder =
      AudioEncoderOpus::MakeAudioEncoder(env, enc_config, {.payload_type = 17});

  // Create decoder.
  AudioDecoderOpus::Config dec_config;
  dec_config.num_channels = kNumChannels;
  auto decoder = AudioDecoderOpus::MakeAudioDecoder(env, dec_config);

  // Open speech file.
  const std::string kInputFileName =
      webrtc::test::ResourcePath("audio_coding/speech_mono_32_48kHz", "pcm");
  test::AudioLoop audio_loop;
  EXPECT_EQ(kSampleRateHz, encoder->SampleRateHz());
  ASSERT_TRUE(audio_loop.Init(kInputFileName, kMaxLoopLengthSamples,
                              kInputBlockSizeSamples));

  EXPECT_LT(EncodedPowerRatio(encoder.get(), decoder.get(), &audio_loop),
            kMaxNarrowbandRatio);

  encoder->OnReceivedTargetAudioBitrate(9000);
  EXPECT_LT(EncodedPowerRatio(encoder.get(), decoder.get(), &audio_loop),
            kMaxNarrowbandRatio);

  encoder->OnReceivedTargetAudioBitrate(9001);
  EXPECT_GT(EncodedPowerRatio(encoder.get(), decoder.get(), &audio_loop),
            kMinWidebandRatio);

  encoder->OnReceivedTargetAudioBitrate(8000);
  EXPECT_GT(EncodedPowerRatio(encoder.get(), decoder.get(), &audio_loop),
            kMinWidebandRatio);

  encoder->OnReceivedTargetAudioBitrate(12001);
  EXPECT_GT(EncodedPowerRatio(encoder.get(), decoder.get(), &audio_loop),
            kMinWidebandRatio);
}

}  // namespace webrtc
