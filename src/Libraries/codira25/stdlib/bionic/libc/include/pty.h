/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
 * @file pty.h
 * @brief Pseudoterminal functions.
 */

#include <sys/cdefs.h>

#include <termios.h>
#include <sys/ioctl.h>

__BEGIN_DECLS

/**
 * [openpty(3)](https://man7.org/linux/man-pages/man3/openpty.3.html) finds
 * a free pseudoterminal and configures it with the given terminal and window
 * size settings.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
int openpty(int* _Nonnull __pty_fd, int* _Nonnull __tty_fd, char* _Nullable __tty_name, const struct termios* _Nullable __termios_ptr, const struct winsize* _Nullable __winsize_ptr) __INTRODUCED_IN(23);

/**
 * [forkpty(3)](https://man7.org/linux/man-pages/man3/forkpty.3.html) creates
 * a new process connected to a pseudoterminal from openpty().
 *
 * Returns 0 in the child/the pid of the child in the parent on success,
 * and returns -1 and sets `errno` on failure.
 *
 * Available since API level 23.
 */
int forkpty(int* _Nonnull __parent_pty_fd, char* _Nullable __child_tty_name, const struct termios* _Nullable __termios_ptr, const struct winsize* _Nullable __winsize_ptr) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


__END_DECLS
