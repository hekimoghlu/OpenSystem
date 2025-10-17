/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
 * @file poll.h
 * @brief Wait for events on a set of file descriptors.
 */

#include <sys/cdefs.h>
#include <linux/poll.h>
#include <signal.h> /* For sigset_t. */
#include <time.h> /* For timespec. */

__BEGIN_DECLS

/** The type of a file descriptor count, used by poll() and ppoll(). */
typedef unsigned int nfds_t;

/**
 * [poll(3)](https://man7.org/linux/man-pages/man3/poll.3.html) waits on a set of file descriptors.
 *
 * Returns the number of ready file descriptors on success, 0 for timeout,
 * and returns -1 and sets `errno` on failure.
 */
int poll(struct pollfd* _Nullable __fds, nfds_t __count, int __timeout_ms);

/**
 * [ppoll(3)](https://man7.org/linux/man-pages/man3/ppoll.3.html) waits on a set of file descriptors
 * or a signal. Set `__timeout` to null for no timeout. Set `__mask` to null to not set the signal
 * mask.
 *
 * Returns the number of ready file descriptors on success, 0 for timeout,
 * and returns -1 and sets `errno` on failure.
 */
int ppoll(struct pollfd* _Nullable __fds, nfds_t __count, const struct timespec* _Nullable __timeout, const sigset_t* _Nullable __mask);

/**
 * Like ppoll() but allows setting a signal mask with RT signals even from a 32-bit process.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
int ppoll64(struct pollfd* _Nullable  __fds, nfds_t __count, const struct timespec* _Nullable __timeout, const sigset64_t* _Nullable __mask) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


#if defined(__BIONIC_INCLUDE_FORTIFY_HEADERS)
#define _POLL_H_
#include <bits/fortify/poll.h>
#undef _POLL_H_
#endif

__END_DECLS
