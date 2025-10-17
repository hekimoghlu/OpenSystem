/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#pragma once

/**
 * @file utmp.h
 * @brief No-op implementation of non-POSIX login records. See <utmpx.h> for the POSIX equivalents.
 */

#include <sys/cdefs.h>
#include <sys/types.h>
#include <time.h>

#define _PATH_UTMP      "/var/run/utmp"
#define _PATH_WTMP      "/var/log/wtmp"
#define _PATH_LASTLOG   "/var/log/lastlog"

#ifdef __LP64__
#define UT_NAMESIZE 32
#define UT_LINESIZE 32
#define UT_HOSTSIZE 256
#else
#define UT_NAMESIZE 8
#define UT_LINESIZE 8
#define UT_HOSTSIZE 16
#endif

#define EMPTY         0
#define RUN_LVL       1
#define BOOT_TIME     2
#define NEW_TIME      3
#define OLD_TIME      4
#define INIT_PROCESS  5
#define LOGIN_PROCESS 6
#define USER_PROCESS  7
#define DEAD_PROCESS  8
#define ACCOUNTING    9

struct lastlog {
  time_t ll_time;
  char ll_line[UT_LINESIZE];
  char ll_host[UT_HOSTSIZE];
};

struct exit_status {
  short e_termination;
  short e_exit;
};

struct utmp {
  short ut_type;
  pid_t ut_pid;
  char ut_line[UT_LINESIZE];
  char ut_id[4];
  char ut_user[UT_NAMESIZE];
  char ut_host[UT_HOSTSIZE];

  struct exit_status ut_exit;

  long ut_session;
  struct timeval ut_tv;

  int32_t ut_addr_v6[4];
  char unused[20];
};

#define ut_name ut_user
#define ut_time ut_tv.tv_sec
#define ut_addr ut_addr_v6[0]

__BEGIN_DECLS

/**
 * Returns -1 and sets errno to ENOTSUP.
 */
int utmpname(const char* _Nonnull __path);

/**
 * Does nothing.
 */
void setutent(void);

/**
 * Does nothing and returns null.
 */
struct utmp* _Nullable getutent(void);

/**
 * Does nothing and returns null.
 */
struct utmp* _Nullable pututline(const struct utmp* _Nonnull __entry);

/**
 * Does nothing.
 */
void endutent(void);

/**
 * [login_tty(3)](https://www.man7.org/linux/man-pages/man3/login_tty.3.html)
 * prepares for login on the given file descriptor.
 *
 * See also forkpty() which combines openpty(), fork(), and login_tty().
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
int login_tty(int __fd) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


__END_DECLS
