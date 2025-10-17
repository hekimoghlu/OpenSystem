/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
 * @file sys/pidfd.h
 * @brief File descriptors representing processes.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

#include <bits/signal_types.h>

__BEGIN_DECLS

/**
 * [pidfd_open(2)](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)
 * opens a file descriptor that refers to a process. This file descriptor will
 * have the close-on-exec flag set by default.
 *
 * Returns a new file descriptor on success and returns -1 and sets `errno` on
 * failure.
 *
 * Available since API level 31.
 */

#if __BIONIC_AVAILABILITY_GUARD(31)
int pidfd_open(pid_t __pid, unsigned int __flags) __INTRODUCED_IN(31);

/**
 * [pidfd_getfd(2)](https://man7.org/linux/man-pages/man2/pidfd_getfd.2.html)
 * dups a file descriptor from another process. This file descriptor will have
 * the close-on-exec flag set by default.
 *
 * Returns a new file descriptor on success and returns -1 and sets `errno` on
 * failure.
 *
 * Available since API level 31.
 */
int pidfd_getfd(int __pidfd, int __targetfd, unsigned int __flags) __INTRODUCED_IN(31);

/**
 * [pidfd_send_signal(2)](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)
 * sends a signal to another process.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 *
 * Available since API level 31.
 */
int pidfd_send_signal(int __pidfd, int __sig, siginfo_t * _Nullable __info, unsigned int __flags) __INTRODUCED_IN(31);
#endif /* __BIONIC_AVAILABILITY_GUARD(31) */


__END_DECLS
