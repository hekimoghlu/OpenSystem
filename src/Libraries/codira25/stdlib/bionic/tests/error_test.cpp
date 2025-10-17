/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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

#include <error.h>

#include <android-base/silent_death_test.h>

using error_DeathTest = SilentDeathTest;

static size_t g_test_error_print_progname_invocation_count;

static void test_error_print_progname() {
  ++g_test_error_print_progname_invocation_count;
}

TEST(error, smoke) {
  error_message_count = 0;
  error(0, 0, "oops");
  ASSERT_EQ(1U, error_message_count);
  error(0, ENOENT, "couldn't open file '%s'", "blah");
  ASSERT_EQ(2U, error_message_count);

  error_print_progname = test_error_print_progname;
  g_test_error_print_progname_invocation_count = 0;
  error(0, 0, "oops");
  ASSERT_EQ(1U, g_test_error_print_progname_invocation_count);

  error_at_line(0, 0, "blah.c", 123, "hello %s", "world");

  error_print_progname = nullptr;
}

TEST(error_DeathTest, error_exit) {
  ASSERT_EXIT(error(22, 0, "x%c", 'y'), ::testing::ExitedWithCode(22), "xy");
}

TEST(error_DeathTest, error_exit_with_errno) {
  ASSERT_EXIT(error(22, EBADF, "x%c", 'y'), ::testing::ExitedWithCode(22), ": xy: Bad file descriptor");
}

TEST(error_DeathTest, error_at_line_exit) {
  ASSERT_EXIT(error_at_line(22, 0, "a.c", 123, "x%c", 'y'), ::testing::ExitedWithCode(22), ":a.c:123: xy");
}

TEST(error_DeathTest, error_at_line_exit_with_errno) {
  ASSERT_EXIT(error_at_line(22, EBADF, "a.c", 123, "x%c", 'y'), ::testing::ExitedWithCode(22), ":a.c:123: xy: Bad file descriptor");
}
