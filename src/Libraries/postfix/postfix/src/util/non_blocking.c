/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
/* System interfaces. */

#include "sys_defs.h"
#include <fcntl.h>

/* Utility library. */

#include "msg.h"
#include "iostuff.h"

/* Backwards compatibility */
#ifndef O_NONBLOCK
#define PATTERN	FNDELAY
#else
#define PATTERN	O_NONBLOCK
#endif

/* non_blocking - set/clear non-blocking flag */

int     non_blocking(fd, on)
int     fd;
int     on;
{
    int     flags;

    if ((flags = fcntl(fd, F_GETFL, 0)) < 0)
	msg_fatal("fcntl: get flags: %m");
    if (fcntl(fd, F_SETFL, on ? flags | PATTERN : flags & ~PATTERN) < 0)
	msg_fatal("fcntl: set non-blocking flag %s: %m", on ? "on" : "off");
    return ((flags & PATTERN) != 0);
}
