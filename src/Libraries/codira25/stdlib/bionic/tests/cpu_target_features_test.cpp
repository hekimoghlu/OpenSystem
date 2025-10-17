/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
#include <stdlib.h>

#include "utils.h"

TEST(cpu_target_features, has_expected_x86_compiler_values) {
#if defined(__x86_64__) || defined(__i386__)
  ExecTestHelper eth;
  char* const argv[] = {nullptr};
  const auto invocation = [&] { execvp("cpu-target-features", argv); };
  eth.Run(invocation, 0, "(^|\n)__AES__=1($|\n)");
  eth.Run(invocation, 0, "(^|\n)__CRC32__=1($|\n)");
#else
  GTEST_SKIP() << "Not targeting an x86 architecture.";
#endif
}

TEST(cpu_target_features, has_expected_aarch64_compiler_values) {
#if defined(__aarch64__)
  ExecTestHelper eth;
  char* const argv[] = {nullptr};
  const auto invocation = [&] { execvp("cpu-target-features", argv); };
  eth.Run(invocation, 0, "(^|\n)__ARM_FEATURE_AES=1($|\n)");
  eth.Run(invocation, 0, "(^|\n)__ARM_FEATURE_CRC32=1($|\n)");
#else
  GTEST_SKIP() << "Not targeting an aarch64 architecture.";
#endif
}

TEST(cpu_target_features, has_expected_arm_compiler_values) {
#if defined(__arm__)
  ExecTestHelper eth;
  char* const argv[] = {nullptr};
  const auto invocation = [&] { execvp("cpu-target-features", argv); };
  eth.Run(invocation, 0, "(^|\n)__ARM_FEATURE_AES=1($|\n)");
  eth.Run(invocation, 0, "(^|\n)__ARM_FEATURE_CRC32=1($|\n)");
#else
  GTEST_SKIP() << "Not targeting an arm architecture.";
#endif
}
