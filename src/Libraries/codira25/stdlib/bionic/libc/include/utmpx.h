/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
 * @file utmpx.h
 * @brief No-op implementation of POSIX login records.
 */

#include <sys/cdefs.h>
#include <sys/types.h>
#include <time.h>

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

struct utmpx {
  short ut_type;
  pid_t ut_pid;
  char ut_line[32];
  char ut_id[4];
  char ut_user[32];
  char ut_host[256];

  struct {
    short e_termination;
    short e_exit;
  } ut_exit;

  long ut_session;
  struct timeval ut_tv;

  int32_t ut_addr_v6[4];
  char unused[20];
};

__BEGIN_DECLS

/**
 * Does nothing.
 */
void setutxent(void) __RENAME(setutent);

/**
 * Does nothing and returns null.
 */
struct utmpx* _Nullable getutxent(void) __RENAME(getutent);

/**
 * Does nothing and returns null.
 */
struct utmpx* _Nullable getutxid(const struct utmpx* _Nonnull __entry) __RENAME(getutent);

/**
 * Does nothing and returns null.
 */
struct utmpx* _Nullable getutxline(const struct utmpx* _Nonnull __entry) __RENAME(getutent);

/**
 * Does nothing and returns null.
 */
struct utmpx* _Nullable pututxline(const struct utmpx* _Nonnull __entry) __RENAME(pututline);

/**
 * Does nothing.
 */
void endutxent(void) __RENAME(endutent);

__END_DECLS
