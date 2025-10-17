/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
 * @file sys/timerfd.h
 * @brief Timer file descriptors.
 */

#include <sys/cdefs.h>

#include <fcntl.h>
#include <linux/timerfd.h>
#include <time.h>
#include <sys/types.h>

__BEGIN_DECLS

/*! \macro TFD_CLOEXEC
 * The timerfd_create() flag for a close-on-exec file descriptor.
 */
/*! \macro TFD_NONBLOCK
 * The timerfd_create() flag for a non-blocking file descriptor.
 */

/**
 * [timerfd_create(2)](https://man7.org/linux/man-pages/man2/timerfd_create.2.html) creates a
 * timer file descriptor.
 *
 * Returns the new file descriptor on success, and returns -1 and sets `errno` on failure.
 */
int timerfd_create(clockid_t __clock, int __flags);

/** The timerfd_settime() flag to use absolute rather than relative times. */
#define TFD_TIMER_ABSTIME (1 << 0)
/** The timerfd_settime() flag to cancel an absolute timer if the realtime clock changes. */
#define TFD_TIMER_CANCEL_ON_SET (1 << 1)

/**
 * [timerfd_settime(2)](https://man7.org/linux/man-pages/man2/timerfd_settime.2.html) starts or
 * stops a timer.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int timerfd_settime(int __fd, int __flags, const struct itimerspec* _Nonnull __new_value, struct itimerspec* _Nullable __old_value);

/**
 * [timerfd_gettime(2)](https://man7.org/linux/man-pages/man2/timerfd_gettime.2.html) queries the
 * current timer settings.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int timerfd_gettime(int __fd, struct itimerspec* _Nonnull __current_value);

__END_DECLS
