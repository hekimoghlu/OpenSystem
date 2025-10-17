/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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

#include <sys/select.h>

#include "header_checks.h"

static void sys_select_h() {
  TYPE(struct timeval);
  STRUCT_MEMBER(struct timeval, time_t, tv_sec);
  STRUCT_MEMBER(struct timeval, suseconds_t, tv_usec);

  TYPE(time_t);
  TYPE(suseconds_t);

  TYPE(sigset_t);
  TYPE(struct timespec);
  TYPE(fd_set);

  MACRO(FD_SETSIZE);

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

  FUNCTION(pselect, int (*f)(int, fd_set*, fd_set*, fd_set*, const struct timespec*, const sigset_t*));
  FUNCTION(select, int (*f)(int, fd_set*, fd_set*, fd_set*, struct timeval*));
}
