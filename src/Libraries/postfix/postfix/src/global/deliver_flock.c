/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#include <unistd.h>

/* Utility library. */

#include <vstring.h>
#include <myflock.h>
#include <iostuff.h>

/* Global library. */

#include "mail_params.h"
#include "deliver_flock.h"

/* Application-specific. */

#define MILLION	1000000

/* deliver_flock - lock open file for mail delivery */

int     deliver_flock(int fd, int lock_style, VSTRING *why)
{
    int     i;

    for (i = 1; /* void */ ; i++) {
	if (myflock(fd, lock_style,
		    MYFLOCK_OP_EXCLUSIVE | MYFLOCK_OP_NOWAIT) == 0)
	    return (0);
	if (i >= var_flock_tries)
	    break;
	rand_sleep(var_flock_delay * MILLION, var_flock_delay * MILLION / 2);
    }
    if (why)
	vstring_sprintf(why, "unable to lock for exclusive access: %m");
    return (-1);
}
