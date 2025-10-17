/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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
#ifndef MODULES_AUDIO_CODING_ACM2_ACM_RECEIVE_TEST_H_
#define MODULES_AUDIO_CODING_ACM2_ACM_RECEIVE_TEST_H_

#include <stddef.h>  // for size_t

#include <memory>
#include <string>

#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/neteq/neteq.h"
#include "api/scoped_refptr.h"
#include "modules/audio_coding/acm2/acm_resampler.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {
class AudioCodingModule;
class AudioDecoder;

namespace test {
class AudioSink;
class PacketSource;

class AcmReceiveTestOldApi {
 public:
  enum NumOutputChannels : size_t {
    kArbitraryChannels = 0,
    kMonoOutput = 1,
    kStereoOutput = 2,
    kQuadOutput = 4
  };

  AcmReceiveTestOldApi(PacketSource* packet_source,
                       AudioSink* audio_sink,
                       int output_freq_hz,
                       NumOutputChannels exptected_output_channels,
                       rtc::scoped_refptr<AudioDecoderFactory> decoder_factory);
  virtual ~AcmReceiveTestOldApi();

  AcmReceiveTestOldApi(const AcmReceiveTestOldApi&) = delete;
  AcmReceiveTestOldApi& operator=(const AcmReceiveTestOldApi&) = delete;

  // Registers the codecs with default parameters from ACM.
  void RegisterDefaultCodecs();

  // Registers codecs with payload types matching the pre-encoded NetEq test
  // files.
  void RegisterNetEqTestCodecs();

  // Runs the test and returns true if successful.
  void Run();

 protected:
  // Method is called after each block of output audio is received from ACM.
  virtual void AfterGetAudio() {}

  SimulatedClock clock_;
  std::unique_ptr<NetEq> neteq_;
  acm2::ResamplerHelper resampler_helper_;
  PacketSource* packet_source_;
  AudioSink* audio_sink_;
  int output_freq_hz_;
  NumOutputChannels exptected_output_channels_;
};

// This test toggles the output frequency every `toggle_period_ms`. The test
// starts with `output_freq_hz_1`. Except for the toggling, it does the same
// thing as AcmReceiveTestOldApi.
class AcmReceiveTestToggleOutputFreqOldApi : public AcmReceiveTestOldApi {
 public:
  AcmReceiveTestToggleOutputFreqOldApi(
      PacketSource* packet_source,
      AudioSink* audio_sink,
      int output_freq_hz_1,
      int output_freq_hz_2,
      int toggle_period_ms,
      NumOutputChannels exptected_output_channels);

 protected:
  void AfterGetAudio() override;

  const int output_freq_hz_1_;
  const int output_freq_hz_2_;
  const int toggle_period_ms_;
  int64_t last_toggle_time_ms_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_ACM2_ACM_RECEIVE_TEST_H_
