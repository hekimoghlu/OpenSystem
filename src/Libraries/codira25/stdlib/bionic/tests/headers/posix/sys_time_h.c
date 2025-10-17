/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <sys/time.h>

#include "header_checks.h"

static void sys_time_h() {
  TYPE(struct timeval);
  STRUCT_MEMBER(struct timeval, time_t, tv_sec);
  STRUCT_MEMBER(struct timeval, suseconds_t, tv_usec);

  TYPE(struct itimerval);
  STRUCT_MEMBER(struct itimerval, struct timeval, it_interval);
  STRUCT_MEMBER(struct itimerval, struct timeval, it_value);

  TYPE(time_t);
  TYPE(suseconds_t);

  TYPE(fd_set);

  MACRO(ITIMER_REAL);
  MACRO(ITIMER_VIRTUAL);
  MACRO(ITIMER_PROF);

#if !defined(FD_CLR)
#error FD_CLR
#endif
#if !defined(FD_ISSET)
#error FD_ISSET
#endif
#if !defined(FD_SET)
#error FD_SET
#endif
#if !defined(FD_ZERO)
#error FD_ZERO
#endif
  MACRO(FD_SETSIZE);

  FUNCTION(getitimer, int (*f)(int, struct itimerval*));
#if defined(__BIONIC__)
  FUNCTION(gettimeofday, int (*f)(struct timeval*, struct timezone*));
#else
  FUNCTION(gettimeofday, int (*f)(struct timeval*, void*));
#endif
  FUNCTION(setitimer, int (*f)(int, const struct itimerval*, struct itimerval*));
  FUNCTION(select, int (*f)(int, fd_set*, fd_set*, fd_set*, struct timeval*));
  FUNCTION(utimes, int (*f)(const char*, const struct timeval[2]));
}
