/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
#include <gtest/gtest.h>

#if defined(__BIONIC__)
#include "gtest_globals.h"
#include "utils.h"
#endif  // defined(__BIONIC__)

#include <android-base/test_utils.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <tuple>

#include "platform/bionic/mte.h"

class MemtagGlobalsTest : public testing::TestWithParam<bool> {};

TEST_P(MemtagGlobalsTest, test) {
  SKIP_WITH_HWASAN << "MTE globals tests are incompatible with HWASan";
#if defined(__BIONIC__) && defined(__aarch64__)
  SKIP_WITH_NATIVE_BRIDGE;  // http://b/242170715
  std::string binary = GetTestLibRoot() + "/memtag_globals_binary";
  bool is_static = MemtagGlobalsTest::GetParam();
  if (is_static) {
    binary += "_static";
  }

  chmod(binary.c_str(), 0755);
  ExecTestHelper eth;
  eth.SetArgs({binary.c_str(), nullptr});
  eth.Run(
      [&]() {
        execve(binary.c_str(), eth.GetArgs(), eth.GetEnv());
        GTEST_FAIL() << "Failed to execve: " << strerror(errno) << " " << binary.c_str();
      },
      // We catch the global-buffer-overflow and crash only when MTE globals is
      // supported. Note that MTE globals is unsupported for fully static
      // executables, but we should still make sure the binary passes its
      // assertions, just that global variables won't be tagged.
      (mte_supported() && !is_static) ? -SIGSEGV : 0, "Assertions were passed");
#else
  GTEST_SKIP() << "bionic/arm64 only";
#endif
}

INSTANTIATE_TEST_SUITE_P(MemtagGlobalsTest, MemtagGlobalsTest, testing::Bool(),
                         [](const ::testing::TestParamInfo<MemtagGlobalsTest::ParamType>& info) {
                           if (info.param) return "MemtagGlobalsTest_static";
                           return "MemtagGlobalsTest";
                         });

TEST(MemtagGlobalsTest, RelrRegressionTestForb314038442) {
  SKIP_WITH_HWASAN << "MTE globals tests are incompatible with HWASan";
#if defined(__BIONIC__) && defined(__aarch64__)
  std::string binary = GetTestLibRoot() + "/mte_globals_relr_regression_test_b_314038442";
  chmod(binary.c_str(), 0755);
  ExecTestHelper eth;
  eth.SetArgs({binary.c_str(), nullptr});
  eth.Run(
      [&]() {
        execve(binary.c_str(), eth.GetArgs(), eth.GetEnv());
        GTEST_FAIL() << "Failed to execve: " << strerror(errno) << " " << binary.c_str();
      },
      /* exit code */ 0, "Program loaded successfully.*Tags are zero!");
#else
  GTEST_SKIP() << "bionic/arm64 only";
#endif
}

TEST(MemtagGlobalsTest, RelrRegressionTestForb314038442WithMteGlobals) {
  if (!mte_supported()) GTEST_SKIP() << "Must have MTE support.";
#if defined(__BIONIC__) && defined(__aarch64__)
  std::string binary = GetTestLibRoot() + "/mte_globals_relr_regression_test_b_314038442_mte";
  chmod(binary.c_str(), 0755);
  ExecTestHelper eth;
  eth.SetArgs({binary.c_str(), nullptr});
  eth.Run(
      [&]() {
        execve(binary.c_str(), eth.GetArgs(), eth.GetEnv());
        GTEST_FAIL() << "Failed to execve: " << strerror(errno) << " " << binary.c_str();
      },
      /* exit code */ 0, "Program loaded successfully.*Tags are non-zero");
#else
  GTEST_SKIP() << "bionic/arm64 only";
#endif
}
