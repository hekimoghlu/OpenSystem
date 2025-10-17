/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>
#include <warn_stat.h>

/* Global library. */

#include <mail_flow.h>

/* Master library. */

#include <master_proto.h>

#define BUFFER_SIZE	1024

/* mail_flow_get - read N tokens */

ssize_t mail_flow_get(ssize_t len)
{
    const char *myname = "mail_flow_get";
    char    buf[BUFFER_SIZE];
    struct stat st;
    ssize_t count;
    ssize_t n = 0;

    /*
     * Sanity check.
     */
    if (len <= 0)
	msg_panic("%s: bad length %ld", myname, (long) len);

    /*
     * Silence some wild claims.
     */
    if (fstat(MASTER_FLOW_WRITE, &st) < 0)
	msg_fatal("fstat flow pipe write descriptor: %m");

    /*
     * Read and discard N bytes. XXX AIX read() can return 0 when an open
     * pipe is empty.
     */
    for (count = len; count > 0; count -= n)
	if ((n = read(MASTER_FLOW_READ, buf, count > BUFFER_SIZE ?
		      BUFFER_SIZE : count)) <= 0)
	    return (-1);
    if (msg_verbose)
	msg_info("%s: %ld %ld", myname, (long) len, (long) (len - count));
    return (len - count);
}

/* mail_flow_put - put N tokens */

ssize_t mail_flow_put(ssize_t len)
{
    const char *myname = "mail_flow_put";
    char    buf[BUFFER_SIZE];
    ssize_t count;
    ssize_t n = 0;

    /*
     * Sanity check.
     */
    if (len <= 0)
	msg_panic("%s: bad length %ld", myname, (long) len);

    /*
     * Write or discard N bytes.
     */
    memset(buf, 0, len > BUFFER_SIZE ? BUFFER_SIZE : len);

    for (count = len; count > 0; count -= n)
	if ((n = write(MASTER_FLOW_WRITE, buf, count > BUFFER_SIZE ?
		       BUFFER_SIZE : count)) < 0)
	    return (-1);
    if (msg_verbose)
	msg_info("%s: %ld %ld", myname, (long) len, (long) (len - count));
    return (len - count);
}

/* mail_flow_count - return number of available tokens */

ssize_t mail_flow_count(void)
{
    const char *myname = "mail_flow_count";
    ssize_t count;

    if ((count = peekfd(MASTER_FLOW_READ)) < 0)
	msg_warn("%s: %m", myname);
    return (count);
}
