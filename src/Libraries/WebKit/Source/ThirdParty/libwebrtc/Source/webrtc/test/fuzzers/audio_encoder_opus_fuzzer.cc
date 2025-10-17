/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
#include "api/audio_codecs/opus/audio_encoder_opus.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "rtc_base/checks.h"
#include "test/fuzzers/audio_encoder_fuzzer.h"

namespace webrtc {

void FuzzOneInput(const uint8_t* data, size_t size) {
  // Create Environment once because creating it for each input noticably
  // reduces the speed of the fuzzer.
  static const Environment* const env = new Environment(CreateEnvironment());

  AudioEncoderOpus::Config config;
  config.frame_size_ms = 20;
  RTC_CHECK(config.IsOk());

  FuzzAudioEncoder(
      /*data_view=*/{data, size},
      /*encoder=*/AudioEncoderOpus::MakeAudioEncoder(*env, config,
                                                     {.payload_type = 100}));
}

}  // namespace webrtc
