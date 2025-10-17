/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_MOCK_WAVREADER_FACTORY_H_
#define MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_MOCK_WAVREADER_FACTORY_H_

#include <map>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "modules/audio_processing/test/conversational_speech/wavreader_abstract_factory.h"
#include "modules/audio_processing/test/conversational_speech/wavreader_interface.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

class MockWavReaderFactory : public WavReaderAbstractFactory {
 public:
  struct Params {
    int sample_rate;
    size_t num_channels;
    size_t num_samples;
  };

  MockWavReaderFactory(const Params& default_params,
                       const std::map<std::string, const Params>& params);
  explicit MockWavReaderFactory(const Params& default_params);
  ~MockWavReaderFactory();

  MOCK_METHOD(std::unique_ptr<WavReaderInterface>,
              Create,
              (absl::string_view),
              (const, override));

 private:
  // Creates a MockWavReader instance using the parameters in
  // audiotrack_names_params_ if the entry corresponding to filepath exists,
  // otherwise creates a MockWavReader instance using the default parameters.
  std::unique_ptr<WavReaderInterface> CreateMock(absl::string_view filepath);

  const Params& default_params_;
  std::map<std::string, const Params> audiotrack_names_params_;
};

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_MOCK_WAVREADER_FACTORY_H_
