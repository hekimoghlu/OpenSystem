/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#ifndef COMMON_AUDIO_VAD_MOCK_MOCK_VAD_H_
#define COMMON_AUDIO_VAD_MOCK_MOCK_VAD_H_

#include "common_audio/vad/include/vad.h"
#include "test/gmock.h"

namespace webrtc {

class MockVad : public Vad {
 public:
  ~MockVad() override { Die(); }
  MOCK_METHOD(void, Die, ());

  MOCK_METHOD(enum Activity,
              VoiceActivity,
              (const int16_t* audio, size_t num_samples, int sample_rate_hz),
              (override));
  MOCK_METHOD(void, Reset, (), (override));
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_VAD_MOCK_MOCK_VAD_H_
