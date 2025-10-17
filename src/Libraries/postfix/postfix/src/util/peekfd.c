/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
/* System library. */

#include <sys_defs.h>
#include <sys/ioctl.h>
#ifdef FIONREAD_IN_SYS_FILIO_H
#include <sys/filio.h>
#endif
#ifdef FIONREAD_IN_TERMIOS_H
#include <termios.h>
#endif
#include <unistd.h>

#ifndef SHUT_RDWR
#define SHUT_RDWR  2
#endif

/* Utility library. */

#include "iostuff.h"

/* peekfd - return amount of data ready to read */

ssize_t peekfd(int fd)
{

    /*
     * Anticipate a series of system-dependent code fragments.
     */
#ifdef FIONREAD
    int     count;

#ifdef SUNOS5

    /*
     * With Solaris10, write_wait() hangs in poll() until timeout, when
     * invoked after peekfd() has received an ECONNRESET error indication.
     * This happens when a client sends QUIT and closes the connection
     * immediately.
     */
    if (ioctl(fd, FIONREAD, (char *) &count) < 0) {
	(void) shutdown(fd, SHUT_RDWR);
	return (-1);
    } else {
	return (count);
    }
#else						/* SUNOS5 */
    return (ioctl(fd, FIONREAD, (char *) &count) < 0 ? -1 : count);
#endif						/* SUNOS5 */
#else
#error "don't know how to look ahead"
#endif
}
