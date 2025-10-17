/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_WAVREADER_FACTORY_H_
#define MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_WAVREADER_FACTORY_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "modules/audio_processing/test/conversational_speech/wavreader_abstract_factory.h"
#include "modules/audio_processing/test/conversational_speech/wavreader_interface.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

class WavReaderFactory : public WavReaderAbstractFactory {
 public:
  WavReaderFactory();
  ~WavReaderFactory() override;
  std::unique_ptr<WavReaderInterface> Create(
      absl::string_view filepath) const override;
};

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_WAVREADER_FACTORY_H_
