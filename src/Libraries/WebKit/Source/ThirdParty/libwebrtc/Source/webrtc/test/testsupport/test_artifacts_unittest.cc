/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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

#include <string>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "rtc_base/system/file_wrapper.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

ABSL_DECLARE_FLAG(std::string, test_artifacts_dir);

namespace webrtc {
namespace test {

TEST(IsolatedOutputTest, ShouldRejectInvalidIsolatedOutDir) {
  const std::string backup = absl::GetFlag(FLAGS_test_artifacts_dir);
  absl::SetFlag(&FLAGS_test_artifacts_dir, "");
  ASSERT_FALSE(WriteToTestArtifactsDir("a-file", "some-contents"));
  absl::SetFlag(&FLAGS_test_artifacts_dir, backup);
}

TEST(IsolatedOutputTest, ShouldRejectInvalidFileName) {
  ASSERT_FALSE(WriteToTestArtifactsDir(nullptr, "some-contents"));
  ASSERT_FALSE(WriteToTestArtifactsDir("", "some-contents"));
}

// Sets isolated_out_dir=<a-writable-path> to execute this test.
TEST(IsolatedOutputTest, ShouldBeAbleToWriteContent) {
  const char* filename = "a-file";
  const char* content = "some-contents";
  if (WriteToTestArtifactsDir(filename, content)) {
    std::string out_file =
        JoinFilename(absl::GetFlag(FLAGS_test_artifacts_dir), filename);
    FileWrapper input = FileWrapper::OpenReadOnly(out_file);
    EXPECT_TRUE(input.is_open());
    EXPECT_TRUE(input.Rewind());
    uint8_t buffer[32];
    EXPECT_EQ(input.Read(buffer, strlen(content)), strlen(content));
    buffer[strlen(content)] = 0;
    EXPECT_EQ(std::string(content),
              std::string(reinterpret_cast<char*>(buffer)));
    input.Close();

    EXPECT_TRUE(RemoveFile(out_file));
  }
}

}  // namespace test
}  // namespace webrtc
