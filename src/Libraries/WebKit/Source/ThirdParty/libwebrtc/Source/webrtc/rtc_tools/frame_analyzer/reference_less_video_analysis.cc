/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "rtc_tools/frame_analyzer/reference_less_video_analysis_lib.h"

ABSL_FLAG(std::string,
          video_file,
          "",
          "Path of the video file to be analyzed, only y4m file format is "
          "supported");

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage(
      "Outputs the freezing score by comparing "
      "current frame with the previous frame.\n"
      "Example usage:\n"
      "./reference_less_video_analysis "
      "--video_file=video_file.y4m\n");
  absl::ParseCommandLine(argc, argv);

  std::string video_file = absl::GetFlag(FLAGS_video_file);
  if (video_file.empty()) {
    exit(EXIT_FAILURE);
  }

  return run_analysis(video_file);
}
