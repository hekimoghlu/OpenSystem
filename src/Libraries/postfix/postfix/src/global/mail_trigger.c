/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <iostuff.h>
#include <trigger.h>
#include <warn_stat.h>

/* Global library. */

#include "mail_params.h"
#include "mail_proto.h"

/* mail_trigger - trigger a service */

int     mail_trigger(const char *class, const char *service,
		             const char *req_buf, ssize_t req_len)
{
    struct stat st;
    char   *path;
    int     status;

    /*
     * XXX Some systems cannot tell the difference between a named pipe
     * (fifo) or a UNIX-domain socket. So we may have to try both.
     */
    path = mail_pathname(class, service);
    if ((status = stat(path, &st)) < 0) {
	 msg_warn("unable to look up %s: %m", path);
    } else if (S_ISFIFO(st.st_mode)) {
	status = fifo_trigger(path, req_buf, req_len, var_trigger_timeout);
	if (status < 0 && S_ISSOCK(st.st_mode))
	    status = LOCAL_TRIGGER(path, req_buf, req_len, var_trigger_timeout);
    } else if (S_ISSOCK(st.st_mode)) {
	status = LOCAL_TRIGGER(path, req_buf, req_len, var_trigger_timeout);
    } else {
	msg_warn("%s is not a socket or a fifo", path);
	status = -1;
    }
    myfree(path);
    return (status);
}
