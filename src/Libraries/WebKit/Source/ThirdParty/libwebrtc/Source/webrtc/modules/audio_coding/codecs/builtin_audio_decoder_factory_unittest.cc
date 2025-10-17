/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include "api/audio_codecs/builtin_audio_decoder_factory.h"

#include <memory>

#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "test/gtest.h"

namespace webrtc {

TEST(AudioDecoderFactoryTest, CreateUnknownDecoder) {
  const Environment env = CreateEnvironment();
  rtc::scoped_refptr<AudioDecoderFactory> adf =
      CreateBuiltinAudioDecoderFactory();
  ASSERT_TRUE(adf);
  EXPECT_FALSE(adf->Create(env, SdpAudioFormat("rey", 8000, 1), std::nullopt));
}

TEST(AudioDecoderFactoryTest, CreatePcmu) {
  const Environment env = CreateEnvironment();
  rtc::scoped_refptr<AudioDecoderFactory> adf =
      CreateBuiltinAudioDecoderFactory();
  ASSERT_TRUE(adf);
  // PCMu supports 8 kHz, and any number of channels.
  EXPECT_FALSE(adf->Create(env, SdpAudioFormat("pcmu", 8000, 0), std::nullopt));
  EXPECT_TRUE(adf->Create(env, SdpAudioFormat("pcmu", 8000, 1), std::nullopt));
  EXPECT_TRUE(adf->Create(env, SdpAudioFormat("pcmu", 8000, 2), std::nullopt));
  EXPECT_TRUE(adf->Create(env, SdpAudioFormat("pcmu", 8000, 3), std::nullopt));
  EXPECT_FALSE(
      adf->Create(env, SdpAudioFormat("pcmu", 16000, 1), std::nullopt));
}

TEST(AudioDecoderFactoryTest, CreatePcma) {
  const Environment env = CreateEnvironment();
  rtc::scoped_refptr<AudioDecoderFactory> adf =
      CreateBuiltinAudioDecoderFactory();
  ASSERT_TRUE(adf);
  // PCMa supports 8 kHz, and any number of channels.
  EXPECT_FALSE(adf->Create(env, SdpAudioFormat("pcma", 8000, 0), std::nullopt));
  EXPECT_TRUE(adf->Create(env, SdpAudioFormat("pcma", 8000, 1), std::nullopt));
  EXPECT_TRUE(adf->Create(env, SdpAudioFormat("pcma", 8000, 2), std::nullopt));
  EXPECT_TRUE(adf->Create(env, SdpAudioFormat("pcma", 8000, 3), std::nullopt));
  EXPECT_FALSE(
      adf->Create(env, SdpAudioFormat("pcma", 16000, 1), std::nullopt));
}

TEST(AudioDecoderFactoryTest, CreateL16) {
  const Environment env = CreateEnvironment();
  rtc::scoped_refptr<AudioDecoderFactory> adf =
      CreateBuiltinAudioDecoderFactory();
  ASSERT_TRUE(adf);
  // L16 supports any clock rate and any number of channels up to 24.
  const int clockrates[] = {8000, 16000, 32000, 48000};
  const int num_channels[] = {1, 2, 3, 24};
  for (int clockrate : clockrates) {
    EXPECT_FALSE(
        adf->Create(env, SdpAudioFormat("l16", clockrate, 0), std::nullopt));
    for (int channels : num_channels) {
      EXPECT_TRUE(adf->Create(env, SdpAudioFormat("l16", clockrate, channels),
                              std::nullopt));
    }
  }
}

// Tests that using more channels than the maximum does not work
TEST(AudioDecoderFactoryTest, MaxNrOfChannels) {
  const Environment env = CreateEnvironment();
  rtc::scoped_refptr<AudioDecoderFactory> adf =
      CreateBuiltinAudioDecoderFactory();
  std::vector<std::string> codecs = {
#ifdef WEBRTC_CODEC_OPUS
      "opus",
#endif
      "pcmu", "pcma", "l16", "G722", "G711",
  };

  for (auto codec : codecs) {
    EXPECT_FALSE(adf->Create(
        env,
        SdpAudioFormat(codec, 32000, AudioDecoder::kMaxNumberOfChannels + 1),
        std::nullopt));
  }
}

TEST(AudioDecoderFactoryTest, CreateG722) {
  const Environment env = CreateEnvironment();
  rtc::scoped_refptr<AudioDecoderFactory> adf =
      CreateBuiltinAudioDecoderFactory();
  ASSERT_TRUE(adf);
  // g722 supports 8 kHz, 1-2 channels.
  EXPECT_FALSE(adf->Create(env, SdpAudioFormat("g722", 8000, 0), std::nullopt));
  EXPECT_TRUE(adf->Create(env, SdpAudioFormat("g722", 8000, 1), std::nullopt));
  EXPECT_TRUE(adf->Create(env, SdpAudioFormat("g722", 8000, 2), std::nullopt));
  EXPECT_FALSE(adf->Create(env, SdpAudioFormat("g722", 8000, 3), std::nullopt));
  EXPECT_FALSE(
      adf->Create(env, SdpAudioFormat("g722", 16000, 1), std::nullopt));
  EXPECT_FALSE(
      adf->Create(env, SdpAudioFormat("g722", 32000, 1), std::nullopt));

  // g722 actually uses a 16 kHz sample rate instead of the nominal 8 kHz.
  std::unique_ptr<AudioDecoder> dec =
      adf->Create(env, SdpAudioFormat("g722", 8000, 1), std::nullopt);
  EXPECT_EQ(16000, dec->SampleRateHz());
}

TEST(AudioDecoderFactoryTest, CreateOpus) {
  const Environment env = CreateEnvironment();
  rtc::scoped_refptr<AudioDecoderFactory> adf =
      CreateBuiltinAudioDecoderFactory();
  ASSERT_TRUE(adf);
  // Opus supports 48 kHz, 2 channels, and wants a "stereo" parameter whose
  // value is either "0" or "1".
  for (int hz : {8000, 16000, 32000, 48000}) {
    for (int channels : {0, 1, 2, 3}) {
      for (std::string stereo : {"XX", "0", "1", "2"}) {
        CodecParameterMap params;
        if (stereo != "XX") {
          params["stereo"] = stereo;
        }
        const bool good = (hz == 48000 && channels == 2 &&
                           (stereo == "XX" || stereo == "0" || stereo == "1"));
        EXPECT_EQ(
            good,
            static_cast<bool>(adf->Create(
                env, SdpAudioFormat("opus", hz, channels, std::move(params)),
                std::nullopt)));
      }
    }
  }
}

}  // namespace webrtc
