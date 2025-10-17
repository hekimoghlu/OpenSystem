/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#include <iostream>
#include <memory>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "modules/audio_processing/test/conversational_speech/config.h"
#include "modules/audio_processing/test/conversational_speech/multiend_call.h"
#include "modules/audio_processing/test/conversational_speech/simulator.h"
#include "modules/audio_processing/test/conversational_speech/timing.h"
#include "modules/audio_processing/test/conversational_speech/wavreader_factory.h"
#include "test/testsupport/file_utils.h"

ABSL_FLAG(std::string, i, "", "Directory containing the speech turn wav files");
ABSL_FLAG(std::string, t, "", "Path to the timing text file");
ABSL_FLAG(std::string, o, "", "Output wav files destination path");

namespace webrtc {
namespace test {
namespace {

const char kUsageDescription[] =
    "Usage: conversational_speech_generator\n"
    "          -i <path/to/source/audiotracks>\n"
    "          -t <path/to/timing_file.txt>\n"
    "          -o <output/path>\n"
    "\n\n"
    "Command-line tool to generate multiple-end audio tracks to simulate "
    "conversational speech with two or more participants.\n";

}  // namespace

int main(int argc, char* argv[]) {
  std::vector<char*> args = absl::ParseCommandLine(argc, argv);
  if (args.size() != 1) {
    printf("%s", kUsageDescription);
    return 1;
  }
  RTC_CHECK(DirExists(absl::GetFlag(FLAGS_i)));
  RTC_CHECK(FileExists(absl::GetFlag(FLAGS_t)));
  RTC_CHECK(DirExists(absl::GetFlag(FLAGS_o)));

  conversational_speech::Config config(
      absl::GetFlag(FLAGS_i), absl::GetFlag(FLAGS_t), absl::GetFlag(FLAGS_o));

  // Load timing.
  std::vector<conversational_speech::Turn> timing =
      conversational_speech::LoadTiming(config.timing_filepath());

  // Parse timing and audio tracks.
  auto wavreader_factory =
      std::make_unique<conversational_speech::WavReaderFactory>();
  conversational_speech::MultiEndCall multiend_call(
      timing, config.audiotracks_path(), std::move(wavreader_factory));

  // Generate output audio tracks.
  auto generated_audiotrack_pairs =
      conversational_speech::Simulate(multiend_call, config.output_path());

  // Show paths to created audio tracks.
  std::cout << "Output files:" << std::endl;
  for (const auto& output_paths_entry : *generated_audiotrack_pairs) {
    std::cout << "  speaker: " << output_paths_entry.first << std::endl;
    std::cout << "    near end: " << output_paths_entry.second.near_end
              << std::endl;
    std::cout << "    far end: " << output_paths_entry.second.far_end
              << std::endl;
  }

  return 0;
}

}  // namespace test
}  // namespace webrtc

int main(int argc, char* argv[]) {
  return webrtc::test::main(argc, argv);
}
