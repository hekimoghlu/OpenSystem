/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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

#include <poll.h>

#include "header_checks.h"

static void poll_h() {
  TYPE(struct pollfd);
  STRUCT_MEMBER(struct pollfd, int, fd);
  STRUCT_MEMBER(struct pollfd, short, events);
  STRUCT_MEMBER(struct pollfd, short, revents);

#if !defined(__GLIBC__)  // Our glibc is too old.
  TYPE(sigset_t);
  TYPE(struct timespec);
#endif

  TYPE(nfds_t);

  MACRO(POLLIN);
  MACRO(POLLRDNORM);
  MACRO(POLLRDBAND);
  MACRO(POLLPRI);
  MACRO(POLLOUT);
  MACRO(POLLWRNORM);
  MACRO(POLLWRBAND);
  MACRO(POLLERR);
  MACRO(POLLHUP);
  MACRO(POLLNVAL);

  FUNCTION(poll, int (*f)(struct pollfd[], nfds_t, int));
#if !defined(__GLIBC__)  // Our glibc is too old.
  FUNCTION(ppoll, int (*f)(struct pollfd[], nfds_t, const struct timespec*, const sigset_t*));
#endif
}
