/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_DTMF_TONE_GENERATOR_H_
#define MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_DTMF_TONE_GENERATOR_H_

#include "modules/audio_coding/neteq/dtmf_tone_generator.h"
#include "test/gmock.h"

namespace webrtc {

class MockDtmfToneGenerator : public DtmfToneGenerator {
 public:
  ~MockDtmfToneGenerator() override { Die(); }
  MOCK_METHOD(void, Die, ());
  MOCK_METHOD(int, Init, (int fs, int event, int attenuation), (override));
  MOCK_METHOD(void, Reset, (), (override));
  MOCK_METHOD(int,
              Generate,
              (size_t num_samples, AudioMultiVector* output),
              (override));
  MOCK_METHOD(bool, initialized, (), (const, override));
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_DTMF_TONE_GENERATOR_H_
