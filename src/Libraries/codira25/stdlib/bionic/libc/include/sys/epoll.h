/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
 * @file sys/epoll.h
 * @brief I/O event file descriptors.
 */

#include <sys/cdefs.h>
#include <sys/types.h>
#include <signal.h> /* For sigset_t. */

#include <linux/eventpoll.h>

__BEGIN_DECLS

/**
 * [epoll_create(2)](https://man7.org/linux/man-pages/man2/epoll_create.2.html)
 * creates a new [epoll](https://man7.org/linux/man-pages/man7/epoll.7.html)
 * file descriptor.
 *
 * Returns a new file descriptor on success and returns -1 and sets `errno` on
 * failure.
 */
int epoll_create(int __size);

/**
 * [epoll_create1(2)](https://man7.org/linux/man-pages/man2/epoll_create1.2.html)
 * creates a new [epoll](https://man7.org/linux/man-pages/man7/epoll.7.html)
 * file descriptor with the given flags.
 *
 * Returns a new file descriptor on success and returns -1 and sets `errno` on
 * failure.
 */
int epoll_create1(int __flags);

/**
 * [epoll_ctl(2)](https://man7.org/linux/man-pages/man2/epoll_ctl.2.html)
 * adds/modifies/removes file descriptors from the given epoll file descriptor.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int epoll_ctl(int __epoll_fd, int __op, int __fd, struct epoll_event* __BIONIC_COMPLICATED_NULLNESS __event);

/**
 * [epoll_wait(2)](https://man7.org/linux/man-pages/man2/epoll_wait.2.html)
 * waits for an event on the given epoll file descriptor.
 *
 * Returns the number of ready file descriptors on success, 0 on timeout,
 * or -1 and sets `errno` on failure.
 */
int epoll_wait(int __epoll_fd, struct epoll_event* _Nonnull __events, int __event_count, int __timeout_ms);

/**
 * Like epoll_wait() but atomically applying the given signal mask.
 */
int epoll_pwait(int __epoll_fd, struct epoll_event* _Nonnull __events, int __event_count, int __timeout_ms, const sigset_t* _Nullable __mask);

/**
 * Like epoll_pwait() but using a 64-bit signal mask even on 32-bit systems.
 *
 * Available since API level 28.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
int epoll_pwait64(int __epoll_fd, struct epoll_event* _Nonnull __events, int __event_count, int __timeout_ms, const sigset64_t* _Nullable __mask) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


/**
 * Like epoll_pwait() but with a `struct timespec` timeout, for nanosecond resolution.
 *
 * Available since API level 35.
 */

#if __BIONIC_AVAILABILITY_GUARD(35)
int epoll_pwait2(int __epoll_fd, struct epoll_event* _Nonnull __events, int __event_count, const struct timespec* _Nullable __timeout, const sigset_t* _Nullable __mask) __INTRODUCED_IN(35);

/**
 * Like epoll_pwait2() but using a 64-bit signal mask even on 32-bit systems.
 *
 * Available since API level 35.
 */
int epoll_pwait2_64(int __epoll_fd, struct epoll_event* _Nonnull __events, int __event_count, const struct timespec* _Nullable __timeout, const sigset64_t* _Nullable __mask) __INTRODUCED_IN(35);
#endif /* __BIONIC_AVAILABILITY_GUARD(35) */


__END_DECLS
