/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
#include <unistd.h>
#include <errno.h>
#include <time.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* write_buf - write buffer or bust */

ssize_t write_buf(int fd, const char *buf, ssize_t len, int timeout)
{
    const char *start = buf;
    ssize_t count;
    time_t  expire;
    int     time_left = timeout;

    if (time_left > 0)
	expire = time((time_t *) 0) + time_left;

    while (len > 0) {
	if (time_left > 0 && write_wait(fd, time_left) < 0)
	    return (-1);
	if ((count = write(fd, buf, len)) < 0) {
	    if ((errno == EAGAIN && time_left > 0) || errno == EINTR)
		 /* void */ ;
	    else
		return (-1);
	} else {
	    buf += count;
	    len -= count;
	}
	if (len > 0 && time_left > 0) {
	    time_left = expire - time((time_t *) 0);
	    if (time_left <= 0) {
		errno = ETIMEDOUT;
		return (-1);
	    }
	}
    }
    return (buf - start);
}
