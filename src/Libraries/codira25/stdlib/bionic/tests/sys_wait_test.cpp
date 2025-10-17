/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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

#include <sys/wait.h>

TEST(sys_wait, waitid) {
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) _exit(66);

  siginfo_t si = {};
  ASSERT_EQ(0, waitid(P_PID, pid, &si, WEXITED));
  ASSERT_EQ(pid, si.si_pid);
  ASSERT_EQ(66, si.si_status);
  ASSERT_EQ(CLD_EXITED, si.si_code);
}

// https://github.com/android/ndk/issues/1878
TEST(sys_wait, macros) {
#if defined(__GLIBC__)
  // glibc before 2016 requires an lvalue.
#else
  ASSERT_FALSE(WIFEXITED(0x7f));
  ASSERT_TRUE(WIFSTOPPED(0x7f));
  ASSERT_FALSE(WIFCONTINUED(0x7f));

  ASSERT_TRUE(WIFEXITED(0x80));
  ASSERT_FALSE(WIFSTOPPED(0x80));
  ASSERT_FALSE(WIFCONTINUED(0x80));

  ASSERT_FALSE(WIFEXITED(0xffff));
  ASSERT_FALSE(WIFSTOPPED(0xffff));
  ASSERT_TRUE(WIFCONTINUED(0xffff));
#endif
}
