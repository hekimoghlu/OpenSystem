/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#ifndef MODULES_AUDIO_CODING_TEST_TESTREDFEC_H_
#define MODULES_AUDIO_CODING_TEST_TESTREDFEC_H_

#include <memory>
#include <string>

#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/environment/environment.h"
#include "api/neteq/neteq.h"
#include "common_audio/vad/include/vad.h"
#include "modules/audio_coding/acm2/acm_resampler.h"
#include "modules/audio_coding/test/Channel.h"
#include "modules/audio_coding/test/PCMFile.h"
#include "test/scoped_key_value_config.h"

namespace webrtc {

class TestRedFec final {
 public:
  explicit TestRedFec();
  ~TestRedFec();

  void Perform();

 private:
  void RegisterSendCodec(const std::unique_ptr<AudioCodingModule>& acm,
                         const SdpAudioFormat& codec_format,
                         std::optional<Vad::Aggressiveness> vad_mode,
                         bool use_red);
  void Run();
  void OpenOutFile(int16_t testNumber);

  test::ScopedKeyValueConfig field_trials_;
  const Environment env_;
  const rtc::scoped_refptr<AudioEncoderFactory> encoder_factory_;
  const rtc::scoped_refptr<AudioDecoderFactory> decoder_factory_;
  std::unique_ptr<AudioCodingModule> _acmA;
  std::unique_ptr<NetEq> _neteq;
  acm2::ResamplerHelper _resampler_helper;

  Channel* _channelA2B;

  PCMFile _inFileA;
  PCMFile _outFileB;
  int16_t _testCntr;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_TEST_TESTREDFEC_H_
