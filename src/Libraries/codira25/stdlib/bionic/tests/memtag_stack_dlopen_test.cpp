/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#include <thread>

#include <dlfcn.h>
#include <stdlib.h>

#include <gtest/gtest.h>

#include <android-base/silent_death_test.h>
#include <android-base/test_utils.h>
#include "mte_utils.h"
#include "utils.h"

#if defined(__BIONIC__)
#include <bionic/mte.h>
#endif

TEST(MemtagStackDlopenTest, DependentBinaryGetsMemtagStack) {
#if defined(__BIONIC__) && defined(__aarch64__)
  if (!mte_enabled()) GTEST_SKIP() << "Test requires MTE.";
  if (is_stack_mte_on())
    GTEST_SKIP() << "Stack MTE needs to be off for this test. Are you running fullmte?";

  std::string path =
      android::base::GetExecutableDirectory() + "/testbinary_depends_on_simple_memtag_stack";
  ExecTestHelper eth;
  std::string ld_library_path = "LD_LIBRARY_PATH=" + android::base::GetExecutableDirectory();
  eth.SetArgs({path.c_str(), nullptr});
  eth.SetEnv({ld_library_path.c_str(), nullptr});
  eth.Run([&]() { execve(path.c_str(), eth.GetArgs(), eth.GetEnv()); }, 0, "RAN");
#else
  GTEST_SKIP() << "requires bionic arm64";
#endif
}

TEST(MemtagStackDlopenTest, DependentBinaryGetsMemtagStack2) {
#if defined(__BIONIC__) && defined(__aarch64__)
  if (!mte_enabled()) GTEST_SKIP() << "Test requires MTE.";
  if (is_stack_mte_on())
    GTEST_SKIP() << "Stack MTE needs to be off for this test. Are you running fullmte?";

  std::string path = android::base::GetExecutableDirectory() +
                     "/testbinary_depends_on_depends_on_simple_memtag_stack";
  ExecTestHelper eth;
  std::string ld_library_path = "LD_LIBRARY_PATH=" + android::base::GetExecutableDirectory();
  eth.SetArgs({path.c_str(), nullptr});
  eth.SetEnv({ld_library_path.c_str(), nullptr});
  eth.Run([&]() { execve(path.c_str(), eth.GetArgs(), eth.GetEnv()); }, 0, "RAN");
#else
  GTEST_SKIP() << "requires bionic arm64";
#endif
}

TEST(MemtagStackDlopenTest, DlopenRemapsStack) {
#if defined(__BIONIC__) && defined(__aarch64__)
  // If this test is failing, look at crash logcat for why the test binary died.
  if (!mte_enabled()) GTEST_SKIP() << "Test requires MTE.";
  if (is_stack_mte_on())
    GTEST_SKIP() << "Stack MTE needs to be off for this test. Are you running fullmte?";

  std::string path =
      android::base::GetExecutableDirectory() + "/testbinary_is_stack_mte_after_dlopen";
  std::string lib_path =
      android::base::GetExecutableDirectory() + "/libtest_simple_memtag_stack.so";
  ExecTestHelper eth;
  std::string ld_library_path = "LD_LIBRARY_PATH=" + android::base::GetExecutableDirectory();
  eth.SetArgs({path.c_str(), lib_path.c_str(), nullptr});
  eth.SetEnv({ld_library_path.c_str(), nullptr});
  eth.Run([&]() { execve(path.c_str(), eth.GetArgs(), eth.GetEnv()); }, 0, "RAN");
#else
  GTEST_SKIP() << "requires bionic arm64";
#endif
}

TEST(MemtagStackDlopenTest, DlopenRemapsStack2) {
#if defined(__BIONIC__) && defined(__aarch64__)
  // If this test is failing, look at crash logcat for why the test binary died.
  if (!mte_enabled()) GTEST_SKIP() << "Test requires MTE.";
  if (is_stack_mte_on())
    GTEST_SKIP() << "Stack MTE needs to be off for this test. Are you running fullmte?";

  std::string path =
      android::base::GetExecutableDirectory() + "/testbinary_is_stack_mte_after_dlopen";
  std::string lib_path =
      android::base::GetExecutableDirectory() + "/libtest_depends_on_simple_memtag_stack.so";
  ExecTestHelper eth;
  std::string ld_library_path = "LD_LIBRARY_PATH=" + android::base::GetExecutableDirectory();
  eth.SetArgs({path.c_str(), lib_path.c_str(), nullptr});
  eth.SetEnv({ld_library_path.c_str(), nullptr});
  eth.Run([&]() { execve(path.c_str(), eth.GetArgs(), eth.GetEnv()); }, 0, "RAN");
#else
  GTEST_SKIP() << "requires bionic arm64";
#endif
}
