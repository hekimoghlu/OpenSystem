/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_TIMING_H_
#define MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_TIMING_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

struct Turn {
  Turn(absl::string_view new_speaker_name,
       absl::string_view new_audiotrack_file_name,
       int new_offset,
       int gain)
      : speaker_name(new_speaker_name),
        audiotrack_file_name(new_audiotrack_file_name),
        offset(new_offset),
        gain(gain) {}
  bool operator==(const Turn& b) const;
  std::string speaker_name;
  std::string audiotrack_file_name;
  int offset;
  int gain;
};

// Loads a list of turns from a file.
std::vector<Turn> LoadTiming(absl::string_view timing_filepath);

// Writes a list of turns into a file.
void SaveTiming(absl::string_view timing_filepath,
                rtc::ArrayView<const Turn> timing);

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_TIMING_H_
