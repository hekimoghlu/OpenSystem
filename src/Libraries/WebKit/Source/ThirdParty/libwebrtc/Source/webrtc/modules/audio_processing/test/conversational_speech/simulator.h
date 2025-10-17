/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_SIMULATOR_H_
#define MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_SIMULATOR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "modules/audio_processing/test/conversational_speech/multiend_call.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

struct SpeakerOutputFilePaths {
  SpeakerOutputFilePaths(absl::string_view new_near_end,
                         absl::string_view new_far_end)
      : near_end(new_near_end), far_end(new_far_end) {}
  // Paths to the near-end and far-end audio track files.
  const std::string near_end;
  const std::string far_end;
};

// Generates the near-end and far-end audio track pairs for each speaker.
std::unique_ptr<std::map<std::string, SpeakerOutputFilePaths>> Simulate(
    const MultiEndCall& multiend_call,
    absl::string_view output_path);

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_SIMULATOR_H_
