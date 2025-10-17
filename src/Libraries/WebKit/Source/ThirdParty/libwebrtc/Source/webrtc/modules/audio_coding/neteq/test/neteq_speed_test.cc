/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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

#include <iostream>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "modules/audio_coding/neteq/tools/neteq_performance_test.h"
#include "rtc_base/checks.h"

// Define command line flags.
ABSL_FLAG(int, runtime_ms, 10000, "Simulated runtime in ms.");
ABSL_FLAG(int, lossrate, 10, "Packet lossrate; drop every N packets.");
ABSL_FLAG(float, drift, 0.1f, "Clockdrift factor.");

int main(int argc, char* argv[]) {
  std::vector<char*> args = absl::ParseCommandLine(argc, argv);
  std::string program_name = args[0];
  std::string usage =
      "Tool for measuring the speed of NetEq.\n"
      "Usage: " +
      program_name +
      " [options]\n\n"
      "  --runtime_ms=N         runtime in ms; default is 10000 ms\n"
      "  --lossrate=N           drop every N packets; default is 10\n"
      "  --drift=F              clockdrift factor between 0.0 and 1.0; "
      "default is 0.1\n";
  if (args.size() != 1) {
    printf("%s", usage.c_str());
    return 1;
  }
  RTC_CHECK_GT(absl::GetFlag(FLAGS_runtime_ms), 0);
  RTC_CHECK_GE(absl::GetFlag(FLAGS_lossrate), 0);
  RTC_CHECK(absl::GetFlag(FLAGS_drift) >= 0.0 &&
            absl::GetFlag(FLAGS_drift) < 1.0);

  int64_t result = webrtc::test::NetEqPerformanceTest::Run(
      absl::GetFlag(FLAGS_runtime_ms), absl::GetFlag(FLAGS_lossrate),
      absl::GetFlag(FLAGS_drift));
  if (result <= 0) {
    std::cout << "There was an error" << std::endl;
    return -1;
  }

  std::cout << "Simulation done" << std::endl;
  std::cout << "Runtime = " << result << " ms" << std::endl;
  return 0;
}
