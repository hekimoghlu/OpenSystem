/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_CONFIG_H_
#define MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_CONFIG_H_

#include <string>

#include "absl/strings/string_view.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

struct Config {
  Config(absl::string_view audiotracks_path,
         absl::string_view timing_filepath,
         absl::string_view output_path)
      : audiotracks_path_(audiotracks_path),
        timing_filepath_(timing_filepath),
        output_path_(output_path) {}

  const std::string& audiotracks_path() const;
  const std::string& timing_filepath() const;
  const std::string& output_path() const;

  const std::string audiotracks_path_;
  const std::string timing_filepath_;
  const std::string output_path_;
};

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_CONVERSATIONAL_SPEECH_CONFIG_H_
