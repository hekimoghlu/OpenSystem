/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
#include "test/testsupport/test_artifacts.h"

#include <string.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "rtc_base/logging.h"
#include "rtc_base/system/file_wrapper.h"
#include "test/testsupport/file_utils.h"

namespace {
const std::string& DefaultArtifactPath() {
  static const std::string path = webrtc::test::OutputPathWithRandomDirectory();
  return path;
}
}  // namespace

ABSL_FLAG(std::string,
          test_artifacts_dir,
          DefaultArtifactPath().c_str(),
          "The output folder where test output should be saved.");

namespace webrtc {
namespace test {

bool GetTestArtifactsDir(std::string* out_dir) {
  if (absl::GetFlag(FLAGS_test_artifacts_dir).empty()) {
    RTC_LOG(LS_WARNING) << "No test_out_dir defined.";
    return false;
  }
  *out_dir = absl::GetFlag(FLAGS_test_artifacts_dir);
  return true;
}

bool WriteToTestArtifactsDir(const char* filename,
                             const uint8_t* buffer,
                             size_t length) {
  if (absl::GetFlag(FLAGS_test_artifacts_dir).empty()) {
    RTC_LOG(LS_WARNING) << "No test_out_dir defined.";
    return false;
  }

  if (filename == nullptr || strlen(filename) == 0) {
    RTC_LOG(LS_WARNING) << "filename must be provided.";
    return false;
  }

  std::string full_path =
      JoinFilename(absl::GetFlag(FLAGS_test_artifacts_dir), filename);
  FileWrapper output = FileWrapper::OpenWriteOnly(full_path);

  RTC_LOG(LS_INFO) << "Writing test artifacts in: " << full_path;

  return output.is_open() && output.Write(buffer, length);
}

bool WriteToTestArtifactsDir(const char* filename, const std::string& content) {
  return WriteToTestArtifactsDir(
      filename, reinterpret_cast<const uint8_t*>(content.c_str()),
      content.length());
}

}  // namespace test
}  // namespace webrtc
