/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
 * @file termios.h
 * @brief General terminal interfaces.
 */

#include <sys/cdefs.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <linux/termios.h>

__BEGIN_DECLS

#if __ANDROID_API__ >= 28
// This file is implemented as static inlines before API level 28.
// Strictly these functions were introduced in API level 21, but there were bugs
// in cfmakeraw() and cfsetspeed() until 28.

/**
 * [cfgetispeed(3)](https://man7.org/linux/man-pages/man3/cfgetispeed.3.html)
 * returns the terminal input baud rate.
 */
speed_t cfgetispeed(const struct termios* _Nonnull __t);

/**
 * [cfgetospeed(3)](https://man7.org/linux/man-pages/man3/cfgetospeed.3.html)
 * returns the terminal output baud rate.
 */
speed_t cfgetospeed(const struct termios* _Nonnull __t);

/**
 * [cfmakeraw(3)](https://man7.org/linux/man-pages/man3/cfmakeraw.3.html)
 * configures the terminal for "raw" mode.
 */
void cfmakeraw(struct termios* _Nonnull __t);

/**
 * [cfsetspeed(3)](https://man7.org/linux/man-pages/man3/cfsetspeed.3.html)
 * sets the terminal input and output baud rate.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int cfsetspeed(struct termios* _Nonnull __t, speed_t __speed);

/**
 * [cfsetispeed(3)](https://man7.org/linux/man-pages/man3/cfsetispeed.3.html)
 * sets the terminal input baud rate.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int cfsetispeed(struct termios* _Nonnull _t, speed_t __speed);

/**
 * [cfsetospeed(3)](https://man7.org/linux/man-pages/man3/cfsetospeed.3.html)
 * sets the terminal output baud rate.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int cfsetospeed(struct termios* _Nonnull __t, speed_t __speed);

/**
 * [tcdrain(3)](https://man7.org/linux/man-pages/man3/tcdrain.3.html)
 * waits until all output has been written.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int tcdrain(int __fd);

/**
 * [tcflow(3)](https://man7.org/linux/man-pages/man3/tcflow.3.html)
 * suspends (`TCOOFF`) or resumes (`TCOON`) output, or transmits a
 * stop (`TCIOFF`) or start (`TCION`) to suspend or resume input.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int tcflow(int __fd, int __action);

/**
 * [tcflush(3)](https://man7.org/linux/man-pages/man3/tcflush.3.html)
 * discards pending input (`TCIFLUSH`), output (`TCOFLUSH`), or
 * both (`TCIOFLUSH`). (In `<stdio.h>` terminology, this is a purge rather
 * than a flush.)
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int tcflush(int __fd, int __queue);

/**
 * [tcgetattr(3)](https://man7.org/linux/man-pages/man3/tcgetattr.3.html)
 * reads the configuration of the given terminal.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int tcgetattr(int __fd, struct termios* _Nonnull __t);

/**
 * [tcgetsid(3)](https://man7.org/linux/man-pages/man3/tcgetsid.3.html)
 * returns the session id corresponding to the given fd.
 *
 * Returns a non-negative session id on success and
 * returns -1 and sets `errno` on failure.
 */
pid_t tcgetsid(int __fd);

/**
 * [tcsendbreak(3)](https://man7.org/linux/man-pages/man3/tcsendbreak.3.html)
 * sends a break.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int tcsendbreak(int __fd, int __duration);

/**
 * [tcsetattr(3)](https://man7.org/linux/man-pages/man3/tcsetattr.3.html)
 * writes the configuration of the given terminal.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int tcsetattr(int __fd, int __optional_actions, const struct termios* _Nonnull __t);

#endif

#if __ANDROID_API__ >= 35
// These two functions were POSIX Issue 8 additions that we can also trivially
// implement as inlines for older OS version.

/**
 * tcgetwinsize(3) gets the window size of the given terminal.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int tcgetwinsize(int __fd, struct winsize* _Nonnull __size);

/**
 * tcsetwinsize(3) sets the window size of the given terminal.
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 */
int tcsetwinsize(int __fd, const struct winsize* _Nonnull __size);
#endif

__END_DECLS

#include <android/legacy_termios_inlines.h>
