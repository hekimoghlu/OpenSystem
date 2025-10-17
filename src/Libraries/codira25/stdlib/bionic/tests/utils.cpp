/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#include "utils.h"

#include <string.h>
#include <syscall.h>

#include <string>

#include <android-base/properties.h>

void RunGwpAsanTest(const char* test_name) {
  ExecTestHelper eh;
  eh.SetEnv({"GWP_ASAN_SAMPLE_RATE=1", "GWP_ASAN_PROCESS_SAMPLING=1", "GWP_ASAN_MAX_ALLOCS=40000",
             nullptr});
  std::string filter_arg = "--gtest_filter=";
  filter_arg += test_name;
  std::string exec(testing::internal::GetArgvs()[0]);
  eh.SetArgs({exec.c_str(), "--gtest_also_run_disabled_tests", filter_arg.c_str(), nullptr});
  eh.Run([&]() { execve(exec.c_str(), eh.GetArgs(), eh.GetEnv()); },
         /* expected_exit_status */ 0,
         // |expected_output_regex|, ensure at least one test ran:
         R"(\[  PASSED  \] [1-9][0-9]* test)");
}

void RunSubtestNoEnv(const char* test_name) {
  ExecTestHelper eh;
  std::string filter_arg = "--gtest_filter=";
  filter_arg += test_name;
  std::string exec(testing::internal::GetArgvs()[0]);
  eh.SetArgs({exec.c_str(), "--gtest_also_run_disabled_tests", filter_arg.c_str(), nullptr});
  eh.Run([&]() { execve(exec.c_str(), eh.GetArgs(), eh.GetEnv()); },
         /* expected_exit_status */ 0,
         // |expected_output_regex|, ensure at least one test ran:
         R"(\[  PASSED  \] [1-9]+0? test)");
}

bool IsLowRamDevice() {
  return android::base::GetBoolProperty("ro.config.low_ram", false) ||
         (android::base::GetBoolProperty("ro.debuggable", false) &&
          android::base::GetBoolProperty("debug.force_low_ram", false));
}

#if defined(__GLIBC__) && __GLIBC_MINOR__ < 30
pid_t gettid() {
  return syscall(__NR_gettid);
}
#endif

void PrintTo(const Errno& e, std::ostream* os) {
  // Prefer EINVAL or whatever, but fall back to strerror() to print
  // "Unknown error 666" for bogus values. Not that I've ever seen one,
  // but we shouldn't be looking at an assertion failure unless something
  // weird has happened!
#if defined(__BIONIC__)
  const char* errno_name = strerrorname_np(e.errno_);
  if (errno_name != nullptr) {
    *os << errno_name;
  } else
#endif
  {
    *os << strerror(e.errno_);
  }
}

int64_t NanoTime() {
  auto t = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now());
  return t.time_since_epoch().count();
}

bool operator==(const Errno& lhs, const Errno& rhs) {
  return lhs.errno_ == rhs.errno_;
}
