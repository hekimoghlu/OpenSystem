/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 2, 2023.
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
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "absl/debugging/failure_signal_handler.h"
#include "absl/debugging/symbolize.h"
#include "absl/flags/parse.h"
#include "test/gmock.h"
#include "test/test_main_lib.h"

namespace {

std::vector<std::string> ReplaceDashesWithUnderscores(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  for (std::string& arg : args) {
    // Only replace arguments that starts with a dash.
    if (!arg.empty() && arg[0] == '-') {
      // Don't replace the 2 first characters.
      auto begin = arg.begin() + 2;
      // Replace dashes on the left of '=' or on all the arg if no '=' is found.
      auto end = std::find(arg.begin(), arg.end(), '=');
      std::replace(begin, end, '-', '_');
    }
  }
  return args;
}

std::vector<char*> VectorOfStringsToVectorOfPointers(
    std::vector<std::string>& input) {
  std::vector<char*> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = &(input[i][0]);
  }
  return output;
}

}  // namespace

int main(int argc, char* argv[]) {
  // Initialize the symbolizer to get a human-readable stack trace
  absl::InitializeSymbolizer(argv[0]);
  testing::InitGoogleMock(&argc, argv);
  // Before parsing the arguments with the absl flag library, any internal '-'
  // characters will be converted to '_' characters to make sure the string is a
  // valid attribute name.
  std::vector<std::string> new_argv = ReplaceDashesWithUnderscores(argc, argv);
  std::vector<char*> raw_new_argv = VectorOfStringsToVectorOfPointers(new_argv);
  absl::ParseCommandLine(argc, &raw_new_argv[0]);

// This absl handler use unsupported features/instructions on Fuchsia
#if !defined(WEBRTC_FUCHSIA)
  absl::FailureSignalHandlerOptions options;
  absl::InstallFailureSignalHandler(options);
#endif

  std::unique_ptr<webrtc::TestMain> main = webrtc::TestMain::Create();
  int err_code = main->Init();
  if (err_code != 0) {
    return err_code;
  }
  return main->Run(argc, argv);
}
