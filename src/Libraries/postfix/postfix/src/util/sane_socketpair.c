/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>

/* Utility library. */

#include "msg.h"
#include "sane_socketpair.h"

/* sane_socketpair - sanitize socketpair() error returns */

int     sane_socketpair(int domain, int type, int protocol, int *result)
{
    static int socketpair_ok_errors[] = {
	EINTR,
	0,
    };
    int     count;
    int     err;
    int     ret;

    /*
     * Solaris socketpair() can fail with EINTR.
     */
    while ((ret = socketpair(domain, type, protocol, result)) < 0) {
	for (count = 0; /* void */ ; count++) {
	    if ((err = socketpair_ok_errors[count]) == 0)
		return (ret);
	    if (errno == err) {
		msg_warn("socketpair: %m (trying again)");
		sleep(1);
		break;
	    }
	}
    }
    return (ret);
}
