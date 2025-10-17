/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#include "rtc_base/system/file_wrapper.h"

#include "rtc_base/checks.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {

TEST(FileWrapper, FileSize) {
  auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string test_name =
      std::string(test_info->test_case_name()) + "_" + test_info->name();
  std::replace(test_name.begin(), test_name.end(), '/', '_');
  const std::string temp_filename =
      test::OutputPathWithRandomDirectory() + test_name;

  // Write
  {
    FileWrapper file = FileWrapper::OpenWriteOnly(temp_filename);
    ASSERT_TRUE(file.is_open());
    EXPECT_EQ(file.FileSize(), 0);

    EXPECT_TRUE(file.Write("foo", 3));
    EXPECT_EQ(file.FileSize(), 3);

    // FileSize() doesn't change the file size.
    EXPECT_EQ(file.FileSize(), 3);

    // FileSize() doesn't move the write position.
    EXPECT_TRUE(file.Write("bar", 3));
    EXPECT_EQ(file.FileSize(), 6);
  }

  // Read
  {
    FileWrapper file = FileWrapper::OpenReadOnly(temp_filename);
    ASSERT_TRUE(file.is_open());
    EXPECT_EQ(file.FileSize(), 6);

    char buf[10];
    size_t bytes_read = file.Read(buf, 3);
    EXPECT_EQ(bytes_read, 3u);
    EXPECT_EQ(memcmp(buf, "foo", 3), 0);

    // FileSize() doesn't move the read position.
    EXPECT_EQ(file.FileSize(), 6);

    // Attempting to read past the end reads what is available
    // and sets the EOF flag.
    bytes_read = file.Read(buf, 5);
    EXPECT_EQ(bytes_read, 3u);
    EXPECT_EQ(memcmp(buf, "bar", 3), 0);
    EXPECT_TRUE(file.ReadEof());
  }

  // Clean up temporary file.
  remove(temp_filename.c_str());
}

}  // namespace webrtc
