/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* timed_write - write with deadline */

ssize_t timed_write(int fd, const void *buf, size_t len,
		            int timeout, void *unused_context)
{
    ssize_t ret;

    /*
     * Wait for a limited amount of time for something to happen. If nothing
     * happens, report an ETIMEDOUT error.
     * 
     * XXX Solaris 8 read() fails with EAGAIN after read-select() returns
     * success. The code below exists just in case their write implementation
     * is equally broken.
     * 
     * This condition may also be found on systems where select() returns
     * success on pipes with less than PIPE_BUF bytes of space, and with
     * badly designed software where multiple writers are fighting for access
     * to the same resource.
     */
    for (;;) {
	if (timeout > 0 && write_wait(fd, timeout) < 0)
	    return (-1);
	if ((ret = write(fd, buf, len)) < 0 && timeout > 0 && errno == EAGAIN) {
	    msg_warn("write() returns EAGAIN on a writable file descriptor!");
	    msg_warn("pausing to avoid going into a tight select/write loop!");
	    sleep(1);
	    continue;
	} else if (ret < 0 && errno == EINTR) {
	    continue;
	} else {
	    return (ret);
	}
    }
}
