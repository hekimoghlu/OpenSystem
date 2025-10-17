/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#include <fstream>
#include <iostream>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {
namespace {

const char* const kErrorMessage = "-Out /path/to/output/file is mandatory";

// Writes fake output intended to be parsed by
// quality_assessment.eval_scores.PolqaScore.
void WriteOutputFile(absl::string_view output_file_path) {
  RTC_CHECK_NE(output_file_path, "");
  std::ofstream out(std::string{output_file_path});
  RTC_CHECK(!out.bad());
  out << "* Fake Polqa output" << std::endl;
  out << "FakeField1\tPolqaScore\tFakeField2" << std::endl;
  out << "FakeValue1\t3.25\tFakeValue2" << std::endl;
  out.close();
}

}  // namespace

int main(int argc, char* argv[]) {
  // Find "-Out" and use its next argument as output file path.
  RTC_CHECK_GE(argc, 3) << kErrorMessage;
  const std::string kSoughtFlagName = "-Out";
  for (int i = 1; i < argc - 1; ++i) {
    if (kSoughtFlagName.compare(argv[i]) == 0) {
      WriteOutputFile(argv[i + 1]);
      return 0;
    }
  }
  RTC_FATAL() << kErrorMessage;
}

}  // namespace test
}  // namespace webrtc

int main(int argc, char* argv[]) {
  return webrtc::test::main(argc, argv);
}
