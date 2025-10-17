/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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

#include "sys_defs.h"
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#ifdef USE_SYS_SELECT_H
#include <sys/select.h>
#endif

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* doze - sleep a while */

void    doze(unsigned delay)
{
    struct timeval tv;

#define MILLION	1000000

    tv.tv_sec = delay / MILLION;
    tv.tv_usec = delay % MILLION;
    while (select(0, (fd_set *) 0, (fd_set *) 0, (fd_set *) 0, &tv) < 0)
	if (errno != EINTR)
	    msg_fatal("doze: select: %m");
}

#ifdef TEST

#include <stdlib.h>

int     main(int argc, char **argv)
{
    unsigned delay;

    if (argc != 2 || (delay = atol(argv[1])) == 0)
	msg_fatal("usage: %s microseconds", argv[0]);
    doze(delay);
    exit(0);
}

#endif
