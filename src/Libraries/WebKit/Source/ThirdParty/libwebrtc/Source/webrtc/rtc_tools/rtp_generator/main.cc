/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#include <stdlib.h>

#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "rtc_tools/rtp_generator/rtp_generator.h"

ABSL_FLAG(std::string, input_config, "", "JSON file with config");
ABSL_FLAG(std::string, output_rtpdump, "", "Where to store the rtpdump");

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage(
      "Generates custom configured rtpdumps for the purpose of testing.\n"
      "Example Usage:\n"
      "./rtp_generator --input_config=sender_config.json\n"
      "                --output_rtpdump=my.rtpdump\n");
  absl::ParseCommandLine(argc, argv);

  const std::string config_path = absl::GetFlag(FLAGS_input_config);
  const std::string rtp_dump_path = absl::GetFlag(FLAGS_output_rtpdump);

  if (rtp_dump_path.empty() || config_path.empty()) {
    return EXIT_FAILURE;
  }

  std::optional<webrtc::RtpGeneratorOptions> options =
      webrtc::ParseRtpGeneratorOptionsFromFile(config_path);
  if (!options.has_value()) {
    return EXIT_FAILURE;
  }

  webrtc::RtpGenerator rtp_generator(*options);
  rtp_generator.GenerateRtpDump(rtp_dump_path);

  return EXIT_SUCCESS;
}
