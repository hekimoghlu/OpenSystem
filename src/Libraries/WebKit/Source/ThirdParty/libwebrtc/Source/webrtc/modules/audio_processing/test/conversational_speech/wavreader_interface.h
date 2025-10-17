/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_WAVREADER_INTERFACE_H_
#define MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_WAVREADER_INTERFACE_H_

#include <stddef.h>

#include "api/array_view.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

class WavReaderInterface {
 public:
  virtual ~WavReaderInterface() = default;

  // Returns the number of samples read.
  virtual size_t ReadFloatSamples(rtc::ArrayView<float> samples) = 0;
  virtual size_t ReadInt16Samples(rtc::ArrayView<int16_t> samples) = 0;

  // Getters.
  virtual int SampleRate() const = 0;
  virtual size_t NumChannels() const = 0;
  virtual size_t NumSamples() const = 0;
};

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_WAVREADER_INTERFACE_H_
