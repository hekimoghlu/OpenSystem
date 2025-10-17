/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_MOCK_WAVREADER_H_
#define MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_MOCK_WAVREADER_H_

#include <cstddef>
#include <string>

#include "api/array_view.h"
#include "modules/audio_processing/test/conversational_speech/wavreader_interface.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

class MockWavReader : public WavReaderInterface {
 public:
  MockWavReader(int sample_rate, size_t num_channels, size_t num_samples);
  ~MockWavReader();

  // TODO(alessiob): use ON_CALL to return random samples if needed.
  MOCK_METHOD(size_t, ReadFloatSamples, (rtc::ArrayView<float>), (override));
  MOCK_METHOD(size_t, ReadInt16Samples, (rtc::ArrayView<int16_t>), (override));

  MOCK_METHOD(int, SampleRate, (), (const, override));
  MOCK_METHOD(size_t, NumChannels, (), (const, override));
  MOCK_METHOD(size_t, NumSamples, (), (const, override));

 private:
  const int sample_rate_;
  const size_t num_channels_;
  const size_t num_samples_;
};

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_MOCK_WAVREADER_H_
