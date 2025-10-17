/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include <android-base/file.h>
#include <android-base/unique_fd.h>

#include "linker_phdr.h"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

using ::android::base::GetExecutableDirectory;
using ::android::base::unique_fd;

namespace {

static std::string GetTestElfPath(const std::string& filename) {
  static std::string test_data_dir = GetExecutableDirectory();

  return test_data_dir + "/" + filename;
}

bool GetPadSegment(const std::string& elf_path) {
  std::string path = GetTestElfPath(elf_path);

  unique_fd fd{TEMP_FAILURE_RETRY(open(path.c_str(), O_CLOEXEC | O_RDWR))};
  EXPECT_GE(fd.get(), 0) << "Failed to open " << path << ": " << strerror(errno);

  struct stat file_stat;
  EXPECT_NE(TEMP_FAILURE_RETRY(fstat(fd.get(), &file_stat)), -1)
        << "Failed to stat " << path << ": " << strerror(errno);

  ElfReader elf_reader;
  EXPECT_TRUE(elf_reader.Read(path.c_str(), fd.get(), 0, file_stat.st_size))
        << "Failed to read ELF file";

  return elf_reader.should_pad_segments();
}

};  // anonymous namespace

TEST(crt_pad_segment, note_absent) {
  if (!page_size_migration_supported()) {
    GTEST_SKIP() << "Kernel does not support page size migration";
  }
  ASSERT_FALSE(GetPadSegment("no_crt_pad_segment.so"));
}

TEST(crt_pad_segment, note_present_and_enabled) {
  if (!page_size_migration_supported()) {
    GTEST_SKIP() << "Kernel does not support page size migration";
  }
  ASSERT_TRUE(GetPadSegment("crt_pad_segment_enabled.so"));
}

TEST(crt_pad_segment, note_present_and_disabled) {
  if (!page_size_migration_supported()) {
    GTEST_SKIP() << "Kernel does not support page size migration";
  }
  ASSERT_FALSE(GetPadSegment("crt_pad_segment_disabled.so"));
}
