/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#include <stddef.h>  // glibc's <syslog.h> breaks without this; musl seems fine.

#define SYSLOG_NAMES
#include <syslog.h>

#include <errno.h>
#include <gtest/gtest.h>

#include "utils.h"

TEST(syslog, syslog_percent_m) {
  ExecTestHelper eth;
  eth.Run(
      [&]() {
        openlog("foo", LOG_PERROR, LOG_AUTH);
        errno = EINVAL;
        syslog(LOG_ERR, "a b c: %m");
        closelog();
        exit(0);
      },
      0, "foo: a b c: Invalid argument\n");
}

TEST(syslog, syslog_empty) {
  ExecTestHelper eth;
  eth.Run(
      [&]() {
        openlog("foo", LOG_PERROR, LOG_AUTH);
        errno = EINVAL;
        syslog(LOG_ERR, "");
        closelog();
        exit(0);
      },
      0, "foo: \n");
}

TEST(syslog, syslog_truncation) {
  ExecTestHelper eth;
  eth.Run(
      [&]() {
        openlog("bar", LOG_PERROR, LOG_AUTH);
        char too_long[2048] = {};
        memset(too_long, 'x', sizeof(too_long) - 1);
        syslog(LOG_ERR, "%s", too_long);
        closelog();
        exit(0);
      },
      0, "bar: x{1023}\n");
}

static int by_name(const CODE* array, const char* name) {
  for (auto c = array; c->c_name != nullptr; c++) {
    if (!strcmp(c->c_name, name)) return c->c_val;
  }
  return -1;
}

static const char* by_value(const CODE* array, int value) {
  for (auto c = array; c->c_name != nullptr; c++) {
    if (c->c_val == value) return c->c_name;
  }
  return nullptr;
}

TEST(syslog, facilitynames) {
  ASSERT_STREQ("auth", by_value(facilitynames, LOG_AUTH));
  ASSERT_STREQ("local7", by_value(facilitynames, LOG_LOCAL7));
  ASSERT_EQ(LOG_AUTH, by_name(facilitynames, "auth"));
  ASSERT_EQ(LOG_LOCAL7, by_name(facilitynames, "local7"));
}

TEST(syslog, prioritynames) {
  ASSERT_STREQ("alert", by_value(prioritynames, LOG_ALERT));
  ASSERT_STREQ("err", by_value(prioritynames, LOG_ERR));
  ASSERT_STREQ("warn", by_value(prioritynames, LOG_WARNING));
  ASSERT_EQ(LOG_ALERT, by_name(prioritynames, "alert"));
  ASSERT_EQ(LOG_ERR, by_name(prioritynames, "err"));
  ASSERT_EQ(LOG_WARNING, by_name(prioritynames, "warn"));
  ASSERT_EQ(LOG_WARNING, by_name(prioritynames, "warning"));
}
